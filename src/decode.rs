use anyhow::{Error as E, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, audio, model};
use rand::{distributions::Distribution, SeedableRng};
use std::sync::mpsc::Receiver;
use tokenizers::Tokenizer;
//use tracing_subscriber::prelude::*;

use model::{Config, Whisper};

use crate::Message;

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
enum Task {
    Transcribe,
    Translate,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

pub struct Decoder {
    model: Whisper,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Whisper,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = token_id(&tokenizer, m::NO_SPEECH_TOKEN)?;
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder.forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                println!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                )
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(segments)
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

pub fn decoder(device: &Device) -> Result<Decoder> {
    let config_filename = "model/config.json";
    let tokenizer_filename = "model/tokenizer.json";
    let weights_filename = "model/model.safetensors";

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let model = Whisper::load(&vb, config)?;

    Decoder::new(
        model,
        tokenizer,
        299792458,
        &device,
        None,
        Some(Task::Transcribe),
        false,
        false,
    )
}

pub fn decode(rx: Receiver<Message>) -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let mut dc = decoder(&device)?;

    let mel_bytes = include_bytes!("melfilters.bytes");
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    let mut notification = notify_rust::Notification::new()
        .summary("dictate")
        .body("...")
        .hint(notify_rust::Hint::Urgency(notify_rust::Urgency::Low))
        //.action("default", "default")
        .show()?;

    // Must keep the clipboard around on linux.
    let mut clipboard = arboard::Clipboard::new()?;

    let mut texts = Vec::new();

    while let Ok(msg) = rx.recv() {
        match msg {
            Message::Chunk(buf) => {
                let mel = audio::pcm_to_mel(&buf, &mel_filters);
                let mel_len = mel.len();
                let mel =
                    Tensor::from_vec(mel, (1, m::N_MELS, mel_len / m::N_MELS), &device).unwrap();

                let segments = dc.run(&mel).unwrap();
                for segment in segments {
                    texts.push(segment.dr.text.trim().to_owned());
                }

                let dictation = texts.join(" ");
                let dictation = dictation.trim();
                notification.body(dictation);
                notification.update();
                clipboard.set_text(dictation)?;
            }
            Message::Done => break,
        }
    }

    notification.wait_for_action(|action| match action {
        // keep the dictation on the clipboard until closed?
        _ => (),
    });

    Ok(())
}
