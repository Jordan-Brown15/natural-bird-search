{
  description = "Python development environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      pythonWithPkgs =
        pkgs.python3.withPackages
        (python-pkgs:
          with python-pkgs; [
            ruff
            ipython
            ipykernel
            jupyter
            notebook
            pip
            # torch
            # torchvision
            urllib3
            flask
            transformers
            diffusers
            accelerate
            datasets
            pillow
            matplotlib
            numpy
            seaborn
            zipp
            sentencepiece
            protobuf
          ]);
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          pythonWithPkgs
          pkgs.uv
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

          # Create a local venv for pip-managed packages (torch etc)
          if [ ! -d .venv ]; then
            echo "Creating venv and installing torch with CUDA support..."
            python -m venv --system-site-packages .venv
            .venv/bin/pip install --quiet \
              torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
              --index-url https://download.pytorch.org/whl/cu124
          fi

          export PATH="$PWD/.venv/bin:$PATH"

          # Register kernel pointing at the venv python
          mkdir -p .jupyter/kernels/nix_python
          cat > .jupyter/kernels/nix_python/kernel.json <<EOF
          {
            "argv": [
              "$PWD/.venv/bin/python",
              "-m",
              "ipykernel_launcher",
              "-f",
              "{connection_file}"
            ],
            "display_name": "Python (Nix + CUDA)",
            "language": "python",
            "metadata": { "debugger": true }
          }
          EOF
          export JUPYTER_PATH="$PWD/.jupyter:$JUPYTER_PATH"

          echo "CUDA available: $(.venv/bin/python -c 'import torch; print(torch.cuda.is_available())')"
        '';
      };
    });
}
