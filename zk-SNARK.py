import numpy as np
import json
import subprocess
import os
import shutil
from hashlib import sha256
from training import SimpleLLM

def to_fixed_int(arr,scale_bits=16,field_mod=None):
    S = 1 << scale_bits
    ints = np.round(arr * S).astype(np.int64)
    if field_mod is not None: ints = np.mod(ints,field_mod)
    return ints,S

def poseidon_hash(arr_flat):
    h = sha256()
    h.update(np.array(arr_flat,dtype=np.int64).tobytes())
    return int.from_bytes(h.digest(),'big')

def export(model,context_token_ids,scale_bits=16,field_mod=None,outprefix='zk'):
    export = {}
    S_total = 1 << scale_bits
    export['scale_bits'] = scale_bits
    export['scale'] = S_total
    export['context'] = context_token_ids
    
    arrays = {
        'token_embedding':model.token_embedding,
        'position_embedding': model.position_embedding,
        'W_q':model.W_q,
        'W_k':model.W_k,
        'W_v':model.W_v,
        'W_o':model.W_o,
        'W_ff1':model.W_ff1,
        'W_ff2':model.W_ff2,
        'W_out':model.W_out
    }
    
    ints = {}
    flat = []
    for k,arr in arrays.items():
        iarr, _ = to_fixed_int(np.array(arr,dtype=float),scale_bits=scale_bits,field_mod=field_mod)
        ints[k] = iarr.tolist()
        flat.extend(iarr.flatten().tolist())
        
    commitment = poseidon_hash(flat)
    
    public = {
        "commitment_to_weights":commitment,
        "claimed_output_token":None
    }
    
    with open(f'{outprefix}_public.json','w') as f: json.dump(public,f)
    with open(f'{outprefix}_witness.json','w') as f: json.dump({'weights_int':ints, 'context_token_ids':context_token_ids},f)
    return public, ints

class CircomInputGenerator:
    """Generates properly formatted inputs for Circom circuits"""
    
    def __init__(self, model, scale_bits=16):
        self.model = model
        self.scale_bits = scale_bits
        self.scale = 1 << scale_bits
        self.vocab_size = model.vocab_size
        self.d_model = model.d_model
        self.seq_len = model.context_length
        
    def generate_circuit_input(self, context_token_ids, output_file='input.json'):
        """
        Generate input.json for Circom witness generation
        Format matches your circuit's expected inputs
        """
        
        token_ids = list(context_token_ids)
    
        # Pad or truncate to match seq_len
        if len(token_ids) < self.seq_len:token_ids = token_ids + [0] * (self.seq_len - len(token_ids))
        elif len(token_ids) > self.seq_len:token_ids = token_ids[:self.seq_len]
    
        print(f"DEBUG: token_ids length = {len(token_ids)}")
        print(f"DEBUG: token_ids = {token_ids}")
    
        # --- Pad position embeddings to seq_len ---
        pos_embeds = np.array(self.model.position_embedding, dtype=float)
        if pos_embeds.shape[0] < self.seq_len:
            padding = np.zeros((self.seq_len - pos_embeds.shape[0], pos_embeds.shape[1]))
            pos_embeds = np.vstack([pos_embeds, padding])
        elif pos_embeds.shape[0] > self.seq_len:pos_embeds = pos_embeds[:self.seq_len, :]
    
        # --- Convert all weights to fixed-point integers ---
        def pad_2d(arr, target_rows, target_cols):
            arr = np.array(arr, dtype=float)
            padded = np.zeros((target_rows, target_cols))
            rows = min(arr.shape[0], target_rows)
            cols = min(arr.shape[1], target_cols)
            padded[:rows, :cols] = arr[:rows, :cols]
            return padded
    
        weights_int = {
            'token_embedding': to_fixed_int(self.model.token_embedding, self.scale_bits)[0].tolist(),
            'position_embedding': to_fixed_int(pos_embeds, self.scale_bits)[0].tolist(),
            'W_q': to_fixed_int(pad_2d(self.model.W_q, self.d_model, self.d_model), self.scale_bits)[0].tolist(),
            'W_k': to_fixed_int(pad_2d(self.model.W_k, self.d_model, self.d_model), self.scale_bits)[0].tolist(),
            'W_v': to_fixed_int(pad_2d(self.model.W_v, self.d_model, self.d_model), self.scale_bits)[0].tolist(),
            'W_o': to_fixed_int(pad_2d(self.model.W_o, self.d_model, self.d_model), self.scale_bits)[0].tolist(),
            'W_ff1': to_fixed_int(pad_2d(self.model.W_ff1, self.d_model, self.d_model * 2), self.scale_bits)[0].tolist(),
            'W_ff2': to_fixed_int(pad_2d(self.model.W_ff2, self.d_model * 2, self.d_model), self.scale_bits)[0].tolist(),
            'W_out': to_fixed_int(pad_2d(self.model.W_out, self.d_model, self.vocab_size), self.scale_bits)[0].tolist()
        }
    
        circuit_input = {
            "token_ids": token_ids,  # This should be a list of exactly seq_len integers
            "token_embedding": weights_int['token_embedding'],
            "position_embedding": weights_int['position_embedding'],
            "W_q": weights_int['W_q'],
            "W_k": weights_int['W_k'],
            "W_v": weights_int['W_v'],
            "W_o": weights_int['W_o'],
            "W_ff1": weights_int['W_ff1'],
            "W_ff2": weights_int['W_ff2'],
            "W_out": weights_int['W_out']
        }
    
        # Verify dimensions before saving
        print(f"\nDimension verification:")
        print(f"  token_ids: {len(circuit_input['token_ids'])} (expected {self.seq_len})")
        print(f"  token_embedding: {len(circuit_input['token_embedding'])}x{len(circuit_input['token_embedding'][0])} (expected {self.vocab_size}x{self.d_model})")
    
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(circuit_input, f, indent=2)
    
        print(f"\n✓ Circuit input generated: {output_file}")
        print(f"  Tokens (padded): {token_ids}")
        print(f"  Scale: 2^{self.scale_bits} = {self.scale}")
    
        return circuit_input, weights_int
    
    def check_circuit_params(self, circuit_file):
        """Extract parameters from Circom circuit file"""
        try:
            with open(circuit_file, 'r') as f:
                content = f.read()
                
            # Look for main component instantiation
            # e.g., "component main = SimpleLLM(10, 8, 4);"
            import re
            match = re.search(r'component\s+main\s*=\s*\w+\((\d+),\s*(\d+),\s*(\d+)\)', content)
            
            if match:
                vocab_size = int(match.group(1))
                d_model = int(match.group(2))
                seq_len = int(match.group(3))
                
                print(f"\n Circuit Parameters Found:")
                print(f"  vocab_size: {vocab_size}")
                print(f"  d_model: {d_model}")
                print(f"  seq_len: {seq_len}")
                
                return vocab_size, d_model, seq_len
        except Exception as e:
            print(f"Warning: Could not parse circuit file: {e}")
            
        return None, None, None

class CompleteProof:
    def __init__(self,circuit_name='zk-SNARK',circuit_dir="circom",scale_bits=19):
        self.circuit_name = circuit_name
        self.circuit_dir = circuit_dir
        self.scale_bits = scale_bits
        self.scale = 1 << scale_bits
        
        self.circuit_file = os.path.join(circuit_dir, f"{circuit_name}.circom")
        self.r1cs_file = os.path.join(circuit_dir, f"{circuit_name}.r1cs")
        self.wasm_file = os.path.join(circuit_dir, f"{circuit_name}_js",f"{circuit_name}.wasm")
        self.witness_gen = os.path.join(circuit_dir, f"{circuit_name}_js","generate_witness.js")
        self.witness_file = os.path.join(circuit_dir,"witness.wtns")
        
        self.pot_0 = os.path.join(circuit_dir,"pot18_0000.ptau")
        self.pot_1 = os.path.join(circuit_dir,"pot18_0001.ptau")
        self.pot_final = os.path.join(circuit_dir,"pot18_final.ptau")
        
        self.zkey_0 = os.path.join(circuit_dir,f"{circuit_name}_0000.zkey")
        self.zkey_final = os.path.join(circuit_dir,f"{circuit_name}_final.zkey")
        self.vkey_file = os.path.join(circuit_dir,"verification_key.json")
        
        self.proof_file = os.path.join(circuit_dir,"proof.json")
        self.public_file = os.path.join(circuit_dir,"public.json")
    
    def _parse_circuit_params(self):
        """Parse circuit file to extract parameters"""
        try:
            with open(self.circuit_file, 'r') as f:
                content = f.read()
            
            import re
            match = re.search(r'component\s+main.*=.*\((\d+),\s*(\d+),\s*(\d+)\)', content)
            if match:
                return int(match.group(1)), int(match.group(2)), int(match.group(3))
        except:
            pass
        return None
        
    def run_command(self,cmd,description,show_output=False):
        print(f"\n→ {description}...")
        result = subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))
    
        if result.returncode == 0:
            print(f"  ✓ Success")
            if show_output and result.stdout: 
                print(f"  {result.stdout.strip()}")
            return True
        else:
            print(f"  ✗ Failed")
            # Always show error details
            if result.stderr: 
                print(f"  Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
            return False

    def compile_circuit(self):
        print("Step 1: Compiling Cicuit")
        if os.path.exists(self.r1cs_file):
            print("Circuit Compiled")
            return True
        
        cmd = f"circom {self.circuit_file} --r1cs --wasm --sym -o {self.circuit_dir}"
        return self.run_command(cmd, "Compiling circuit", show_output=True)
    
    def generate_witness(self,input_file="input.json"):
        print("Step 2: Generating Witness")
        if not os.path.exists(self.wasm_file):
            print("✗ WASM file not found. Compile circuit first.")
            return False
        
        # Debug: check input file
        print(f"  Input file: {input_file}")
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                inp = json.load(f)
            print(f"  Input keys: {list(inp.keys())}")
        else:
            print(f"  ✗ Input file not found: {input_file}")
            return False
        
        cmd = f"node {self.witness_gen} {self.wasm_file} {input_file} {self.witness_file}"
        return self.run_command(cmd, "Generating witness")
    
    def pow_tau(self):
        print("Step 3: Powers of Tau")
        
        if os.path.exists(self.pot_final):
            print("✓ Powers of Tau already completed")
            return True
        
        # Phase 1: Start ceremony
        if not os.path.exists(self.pot_0):
            cmd = f"snarkjs powersoftau new bn128 18 {self.pot_0} -v"
            if not self.run_command(cmd, "Starting Powers of Tau"):
                return False
        
        # Phase 2: Contribute
        if not os.path.exists(self.pot_1):
            cmd = f"snarkjs powersoftau contribute {self.pot_0} {self.pot_1} --name='First contribution' -v -e='random entropy'"
            if not self.run_command(cmd, "Contributing to ceremony"):
                return False
        
        # Phase 3: Prepare for phase 2
        cmd = f"snarkjs powersoftau prepare phase2 {self.pot_1} {self.pot_final} -v"
        return self.run_command(cmd, "Preparing phase 2")
    
    def groth16_setup(self):
        print("Step 4: Groth16 Setup")
        
        vkey_valid = os.path.exists(self.vkey_file) and os.path.getsize(self.vkey_file) > 0
        zkey_valid = os.path.exists(self.zkey_final) and os.path.getsize(self.zkey_final) > 0
    
        if vkey_valid and zkey_valid:
            print("✓ Groth16 already set up")
            return True
    
        # Generate initial zkey (delete if empty)
        if os.path.exists(self.zkey_0) and os.path.getsize(self.zkey_0) == 0:
            print("  Removing empty zkey file...")
            os.remove(self.zkey_0)
    
        if not os.path.exists(self.zkey_0):
            cmd = f"snarkjs groth16 setup {self.r1cs_file} {self.pot_final} {self.zkey_0}"
            if not self.run_command(cmd, "Generating proving key"):
                return False
        
            # Verify it was created properly
            if not os.path.exists(self.zkey_0) or os.path.getsize(self.zkey_0) == 0:
                print("  ✗ Failed to generate valid zkey file")
                return False
            else:
                print("  ✓ Initial zkey already exists")
    
        # Try to contribute to phase 2 (optional for testing)
        if not os.path.exists(self.zkey_final) or os.path.getsize(self.zkey_final) == 0:
            if os.path.exists(self.zkey_final):
                os.remove(self.zkey_final)
            
            cmd = f"snarkjs zkey contribute {self.zkey_0} {self.zkey_final} --name='1st Contributor' -v -e='more random entropy'"
            success = self.run_command(cmd, "Contributing to zkey (optional)", show_output=True)
        
            if not success:
                print("  ⚠ Contribution failed - using initial zkey (fine for testing)")
                shutil.copy(self.zkey_0, self.zkey_final)
                print("  ✓ Copied initial zkey to final zkey")
        else:
            print("  ✓ Final zkey already exists")
    
        # Export verification key
        if not os.path.exists(self.vkey_file) or os.path.getsize(self.vkey_file) == 0:
            cmd = f"snarkjs zkey export verificationkey {self.zkey_final} {self.vkey_file}"
            return self.run_command(cmd, "Exporting verification key")
        else:
            print("  ✓ Verification key already exists")
            return True
    
    def generate_proof(self):
        print("Step 5: Generate Proof")
        cmd = f"snarkjs groth16 prove {self.zkey_final} {self.witness_file} {self.proof_file} {self.public_file}"
        
        if self.run_command(cmd, "Generating proof"):
            # Display proof info
            with open(self.proof_file, 'r') as f:
                proof = json.load(f)
            with open(self.public_file, 'r') as f:
                public_signals = json.load(f)
            
            print(f"\n  Proof generated successfully!")
            print(f"  Proof size: {len(json.dumps(proof))} bytes")
            print(f"  Public signals: {public_signals[:5]}..." if len(public_signals) > 5 else f"  Public signals: {public_signals}")
            return True
        
        return False
    
    def verify_proof(self):
        print("Step 6: Verify Proof")
        cmd = f"snarkjs groth16 verify {self.vkey_file} {self.public_file} {self.proof_file}"
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and "OK" in result.stdout:
            print("\n  ✓✓✓ PROOF VERIFIED SUCCESSFULLY! ✓✓✓")
            return True
        else:
            print("\n  ✗ Proof verification FAILED")
            print(f"  Output: {result.stdout}")
            return False
        
    def export(self):
        print("Step 7: Export Solidity Verifier")
        verifier_sol = os.path.join(self.circuit_dir, "verifier.sol")
        cmd = f"snarkjs zkey export solidityverifier {self.zkey_final} {verifier_sol}"
        
        if self.run_command(cmd, "Exporting Solidity verifier"):
            print(f"\n  Contract saved to: {verifier_sol}")
            print(f"  Deploy this to verify proofs on-chain!")
            return True
        
        return False
    
    def full_workflow(self,model,context_token_ids):
        print("\n" + "="*60)
        print(" COMPLETE ZK-SNARK PROOF WORKFLOW")
        print("="*60)
        
        # Step 0: Generate circuit input
        print("\n" + "="*60)
        print("STEP 0: Generate Circuit Input")
        print("="*60)
        
        generator = CircomInputGenerator(model, self.scale_bits)
        input_file = os.path.join(self.circuit_dir, 'input_fixed.json')
        circuit_input, weights_int = generator.generate_circuit_input(context_token_ids, input_file)
        
        # Compute expected output for verification
        logits, _ = model.foward(context_token_ids)
        expected_output = int(np.argmax(logits))
        print(f"  Expected output token: {expected_output}")
        
        # Execute workflow steps
        steps = [
            (self.compile_circuit, []),
            (self.generate_witness, [input_file]),
            (self.pow_tau, []),
            (self.groth16_setup, []),
            (self.generate_proof, []),
            (self.verify_proof, []),
            (self.export, [])
        ]
        
        for i, (step_func, args) in enumerate(steps, 1):
            if not step_func(*args):
                print(f"\n✗ Workflow failed at step {i}")
                return False
        
        print("\n" + "="*60)
        print(" ✓✓✓ WORKFLOW COMPLETED SUCCESSFULLY! ✓✓✓")
        print("="*60)
        print(f"\nFiles generated:")
        print(f"  Proof: {self.proof_file}")
        print(f"  Public signals: {self.public_file}")
        print(f"  Verification key: {self.vkey_file}")
        print(f"  Solidity verifier: {os.path.join(self.circuit_dir, 'verifier.sol')}")
        
        return True

def main(): 
    print("\n" + "="*60)
    print(" LLM ZK-SNARK PROOF SYSTEM")
    print(" Integrating Circom + SnarkJS")
    print("="*60)
    
    # Initialize model
    print("\n[1] Initializing model...")
    model = SimpleLLM(vocab_size=100, d_model=32, context_length=8)
    prompt = [12, 3, 89, 1]
    
    print(f"\nModel Architecture:")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  d_model: {model.d_model}")
    print(f"  Context length: {model.context_length}")
    print(f"  Parameters: ~{model.vocab_size * model.d_model * 2 + model.d_model ** 2 * 6:,}")
    
    # Test forward pass
    print(f"\n[2] Testing forward pass...")
    logits, _ = model.foward(prompt)
    predicted = np.argmax(logits)
    print(f"  Input: {prompt}")
    print(f"  Predicted token: {predicted}")
    print(f"  Top 3: {np.argsort(logits)[-3:][::-1].tolist()}")
    
    zk_system = CompleteProof(scale_bits=18)
    zk_system.full_workflow(model, prompt)
    

if __name__ == '__main__':
    main()
