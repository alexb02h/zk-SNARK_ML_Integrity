import numpy as np
import json

class SimpleLLM:
    def __init__(self,vocab_size=100,d_model=32,context_length=8):
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.context_length=context_length
        
        np.random.seed(42)
        self.token_embedding = self._quantize(np.random.randn(vocab_size,d_model) * 0.1)
        self.position_embedding = self._quantize(np.random.randn(context_length,d_model) * 0.1)
        
        self.W_q = self._quantize(np.random.randn(d_model,d_model) * 0.1)
        self.W_k = self._quantize(np.random.randn(d_model,d_model) * 0.1)
        self.W_v = self._quantize(np.random.randn(d_model,d_model) * 0.1)
        self.W_o = self._quantize(np.random.randn(d_model,d_model) * 0.1)
        
        self.W_ff1 = self._quantize(np.random.randn(d_model,d_model * 2) * 0.1)
        self.W_ff2 = self._quantize(np.random.randn(d_model * 2,d_model) * 0.1)
        
        self.W_out = self._quantize(np.random.randn(d_model,vocab_size) * 0.1)
        
    def _quantize(self,weights,bits=8):
        scale = (2 ** (bits - 1) - 1)
        quantize = np.clip(np.round(weights * scale), -scale, scale)
        return quantize / scale
    
    def _softmax(self,x,axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def attention(self,x):
        seq_len = x.shape[0]
        
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        scores = (Q @ K.T) / np.sqrt(self.d_model)
        
        mask = np.triu(np.ones((seq_len,seq_len)) * -1e9, k=1)
        scores = scores + mask
        
        attention_weights = self._softmax(scores,axis=-1)
        
        output = attention_weights @ V
        return output @ self.W_o
    
    def feed_foward(self,x):
        hidden = x @ self.W_ff1
        return hidden @ self.W_ff2
    
    def foward(self,token_ids):
        token_ids = np.array(token_ids)
        seq_len = len(token_ids)
        
        intermediate_states = {}
        
        token_embeds = self.token_embedding[token_ids]
        pos_embeds = self.position_embedding[:seq_len]
        x = token_embeds + pos_embeds
        intermediate_states['embeddings'] = x.copy()
        
        attention_output = self.attention(x)
        x = x + attention_output
        intermediate_states['post_attention'] = x.copy()
        
        ff_output = self.feed_foward(x)
        x = x + ff_output
        intermediate_states['post_feedfoward'] = x.copy()
        
        logits = x[-1] @ self.W_out
        intermediate_states['logits'] = x.copy()
        
        return logits, intermediate_states
    
    def generate(self,prompt_ids,max_new_tokens=5):
        current_ids = list(prompt_ids)
        all_states = []
        
        for _ in range(max_new_tokens):
            context = current_ids[-self.context_length:]
            logits,states = self.foward(context)
            next_token = np.argmax(logits)
            current_ids.append(int(next_token))
            all_states.append(states)
            
        return current_ids,all_states
    
    def export_circuits(self, filename='model_weights.json'):
        export_data = {
            'architecture':{
                'vocab_size':self.vocab_size,
                'd_model':self.d_model,
                'context_length':self.context_length
            },
            'weights':{
                'token_embedding':self.token_embedding.tolist(),
                'position_embedding': self.position_embedding.tolist(),
                'W_q':self.W_q.tolist(),
                'W_k':self.W_k.tolist(),
                'W_v':self.W_v.tolist(),
                'W_o':self.W_o.tolist(),
                'W_ff1':self.W_ff1.tolist(),
                'W_ff2':self.W_ff2.tolist(),
                'W_out':self.W_out.tolist()
            }
        }
        with open(filename,'w') as f: json.dump(export_data,f)
        print(f"Model exported to {filename}")
        return export_data
    
if __name__ == '__main__':
    model = SimpleLLM(vocab_size=100,d_model=32,context_length=8)
    prompt = [12,3,89,1]
    
    print(f"Model Architecture:")
    print(f" Vocabulary Size: {model.vocab_size}")
    print(f" Embedded dimension: {model.d_model}")
    print(f" Context length: {model.context_length}")
    print(f" Total parameters: ~{model.vocab_size * model.d_model * 2 + model.d_model ** 2 * 6:,}\n")
    
    print(f"Input token: {prompt}")
    logits,states = model.foward(prompt)
    predicted_token = np.argmax(logits)
    print(f"Predicted next token: {predicted_token}")
    print(f"Top 5 predictions: {np.argsort(logits)[-5:][::-1].tolist()}\n")
    
    print(f"Generating sequence ...")
    generated,all_states = model.generate(prompt,max_new_tokens=5)
    print(f"Generated token ID's {generated}\n")
    
    print("Exporting model weights for zk-SNARK circuit")
    model.export_circuits()
        