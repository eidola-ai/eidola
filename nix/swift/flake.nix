{
  description = "Hermetic Swift 6.2 toolchain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    with flake-utils.lib; eachSystem [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ]
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          swiftVersion = "6.2-RELEASE";

          # Per-system source and build configuration
          swiftConfig = {
            "aarch64-darwin" = {
              src = pkgs.fetchurl {
                url = "https://download.swift.org/swift-6.2-release/xcode/swift-6.2-RELEASE/swift-6.2-RELEASE-osx.pkg";
                sha256 = "0jynn925zgvhdfskf08m73y2xbqp3k9lz0chzpnnsyhca09lmb4y";
              };
              builder = ./build.nix;
            };
            "aarch64-linux" = {
              src = pkgs.fetchurl {
                url = "https://download.swift.org/swift-6.2-release/ubuntu2404-aarch64/swift-6.2-RELEASE/swift-6.2-RELEASE-ubuntu24.04-aarch64.tar.gz";
                sha256 = "1a3v1cmw4mxndjksvdxl31w6nzj5896bnm9c4cmvlfbbxrlqmd7v";
              };
              builder = ./build-linux.nix;
            };
            "x86_64-linux" = {
              src = pkgs.fetchurl {
                url = "https://download.swift.org/swift-6.2-release/ubuntu2404/swift-6.2-RELEASE/swift-6.2-RELEASE-ubuntu24.04.tar.gz";
                sha256 = "0xy03fj74r19qw3781vkcj6gdi92kadlnhclwsax9ahw6yin6gcf";
              };
              builder = ./build-linux.nix;
            };
          }.${system};

          swift = pkgs.callPackage swiftConfig.builder {
            version = swiftVersion;
            src = swiftConfig.src;
          };

        in
        {
          packages = {
            inherit swift;
            default = swift;
          };

          formatter = pkgs.nixpkgs-fmt;
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.isDarwin {
          devShells.default = pkgs.mkShell {
            name = "swift-env";
            buildInputs = [
              swift
              pkgs.apple-sdk_26
            ];
          };
        }
      ) // {
        overlays.default = final: prev: {
          swift = self.packages.${prev.system}.swift;
        };
      };
}
