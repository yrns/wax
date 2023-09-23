{ pkgs ? import <nixpkgs> { } }:
let
  rustify = builtins.getFlake "github:yrns/rustify";
  crateOverrides = rustify.lib.crateOverrides {
    lockFile = ./Cargo.lock;
    inherit pkgs;
  };
in pkgs.mkShell rec {
  # nativeBuildInputs = [ pkgs.libconfig ];
  buildInputs = [
    pkgs.cudatoolkit
    pkgs.cudaPackages.cudnn
    pkgs.linuxPackages_latest.nvidia_x11
  ];
  inputsFrom = [ crateOverrides ];
  LD_LIBRARY_PATH =
    "${pkgs.lib.makeLibraryPath (crateOverrides.buildInputs ++ buildInputs)}";
  # CUDA_ROOT = "/nix/store/8a92d7ym9z3ms6rvcfbbgaw2b7zn3zz8-cudatoolkit-11.8.0";
  CUDA_PATH = "${pkgs.cudatoolkit}";
  # CUDA_INC_PATH = "/nix/store/8a92d7ym9z3ms6rvcfbbgaw2b7zn3zz8-cudatoolkit-11.8.0/include";
}
