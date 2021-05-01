{ pkgs ? import nix/pkgs.nix {} }:

let
  app = pkgs.poetry2nix.mkPoetryEnv {
    projectDir = ./.;
    editablePackageSources.jterritory = ./.;
  };
in pkgs.mkShell { buildInputs = [ app pkgs.niv pkgs.poetry ]; }
