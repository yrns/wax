mod decode;

// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{channel, Receiver, Sender},
        Arc,
    },
    time::Duration,
};

use anyhow::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BufferSize, SampleRate, StreamConfig,
};
use tracing_subscriber::prelude::*;

use decode::*;

pub enum Message {
    Chunk(Vec<f32>),
    Done,
}

pub fn rms<'a>(vals: impl IntoIterator<Item = &'a f32>) -> f32 {
    let mut n = 0;
    let pow_sum = vals.into_iter().fold(0., |acc, v| {
        n += 1;
        acc + v.powi(2)
    });
    (pow_sum / n as f32).sqrt()
}

fn main() -> Result<()> {
    tracing_subscriber::registry().init();

    let (tx, rx): (Sender<_>, Receiver<_>) = channel();

    let listen = std::thread::spawn(move || listen(tx).unwrap());
    let decode = std::thread::spawn(move || decode(rx).unwrap());

    listen.join().unwrap();
    decode.join().unwrap();

    Ok(())
}

pub fn listen(tx: Sender<Message>) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default input device"))?;
    println!("Default input device: {}", device.name()?);

    let config = StreamConfig {
        channels: 1,
        sample_rate: SampleRate(16_000),
        buffer_size: BufferSize::Default,
    };

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let mut buf = Vec::new();
    let mut quiet_samples = 0u32;
    let done = Arc::new(AtomicBool::new(false));
    let done0 = done.clone();

    let data_fn = move |data: &[f32], _: &_| {
        if done.load(Ordering::Relaxed) {
            return;
        }

        let rms = rms(data);
        quiet_samples = if rms < 0.006 {
            quiet_samples.saturating_add(data.len() as u32)
        } else {
            buf.extend_from_slice(data);
            0
        };

        let mut msg = None;
        let qs = quiet_samples as f32 / 16_000f32;
        if qs > 0.5 {
            if qs > 3.0 {
                done.store(true, Ordering::Relaxed);
                msg = Some(Message::Done);
            } else if buf.len() > 0 {
                let mut chunk = Vec::new();
                std::mem::swap(&mut buf, &mut chunk);
                msg = Some(Message::Chunk(chunk));
            };

            if let Some(msg) = msg {
                if let Err(_) = tx.send(msg) {
                    done.store(true, Ordering::Relaxed);
                }
            }
        }
    };

    let stream = device.build_input_stream(&config, data_fn, err_fn, None)?;
    stream.play()?;

    while !done0.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(100));
    }

    Ok(())
}
