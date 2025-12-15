# Zero-Knowledge Proofs for Language Models

A proof-of-concept implementation demonstrating how zero-knowledge Succinct Non-Interactive Arguments of Knowledge (zk-SNARKS) cab ve aookied toi verify Language Model outputs without revealing model weights or input data.

## Paper

This repository includes the paper "Zero-Knowledge Proofs for Machine Learning Integrity" which demonstrates verification of LLM predictions while maintaining data confidentiality.

## Software Dependencies

1. **Node.js** (v16 or higher)
```bash
   # macOS
   brew install node
   
   # Ubuntu/Debian
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
```

2. **Circom 2.0**
 ```bash
     #Install Rust (Required for Circom)
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
     source $HOME/.cargo/env

     git clone https://github.com/iden3/circom.git
     cd circom
     cargo build --release
     cargo install --path circom
 ```

3.**SnarkJS**
```bash
      npm install -g snarkjs
```

4.**Python 3.8**
```bash
    #macOS
    brew install python3

    # Ubuntu/Debian
    sudo apt-get install python3 python3-pip
```

## Installation

**Clone the Repository**
```bash
  git clone https://github.com/alexb02h/zk-SNARK_ML_Integrity.git
  cd zk-SNARK_ML_Integrity
```

## Start
Once you've cloned the repository and cd into the directory run
```bash
  python zk-SNARK.py
```
