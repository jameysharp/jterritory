{ sources ? import ./sources.nix }:

import sources.nixpkgs {
  overlays = [
    (import (sources.poetry2nix + "/overlay.nix"))
  ];
}
