{
  description = "Hermetic Swift 6.2 Flake for macOS ARM64";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    with flake-utils.lib; eachSystem [ "aarch64-darwin" ]
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          swift_src = pkgs.fetchurl {
            url = "https://download.swift.org/swift-6.2-release/xcode/swift-6.2-RELEASE/swift-6.2-RELEASE-osx.pkg";
            sha256 = "0jynn925zgvhdfskf08m73y2xbqp3k9lz0chzpnnsyhca09lmb4y";
          };

          swift = pkgs.callPackage ./build.nix {
            version = "6.2-RELEASE";
            src = swift_src;
          };

        in
        rec {
          packages = {
            inherit swift;
            default = swift;
          };

          devShells.default = pkgs.mkShell {
            name = "swift-env";
            buildInputs = [
              swift
              pkgs.apple-sdk_26
            ];
          };

          formatter = pkgs.nixpkgs-fmt;
        }
      ) // {
        overlays.default = final: prev: {
          swift = self.packages.${prev.system}.swift;
        };
      };
}
