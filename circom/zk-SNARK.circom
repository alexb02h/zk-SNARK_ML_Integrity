pragma circom 2.0.0;

template Multiplier(){
    signal input a;
    signal input b;
    signal output c;
    c <== a * b;
}

template SimpleHashN(n){
    signal input in[n];
    signal output out;

    signal acc[n];
    acc[0] <== in[0];

    for(var i=1; i < n; i++) acc[i] <== acc[i-1] + in[i] * in[i];

    out <== acc[n-1];
}

template DotProduct(n){
    signal input x[n];
    signal input w[n];
    signal output out;

    signal products[n];
    signal sums[n];

    for (var i = 0; i<n; i++) products[i] <== x[i] * w[i];

    sums[0] <== products[0];
    for(var i = 1; i<n; i++) sums[i] <== sums[i-1] + products[i];

    out <== sums[n-1];
}

template MatVecMul(rows,cols){
    signal input matrix[rows][cols];
    signal input vector[cols];
    signal output out[rows];

    component dotProducts[rows];
    for(var i = 0; i < rows; i++){
        dotProducts[i] = DotProduct(cols);
        for(var j = 0; j < cols; j++){
            dotProducts[i].x[j] <== matrix[i][j];
            dotProducts[i].w[j] <== vector[j];
        }
        out[i] <== dotProducts[i].out;
    }
}


template MatMul(rowsA,colsA,colsB){
    signal input A[rowsA][colsA];
    signal input B[colsA][colsB];
    signal output C[rowsA][colsB];

    component dotProducts[rowsA][colsB];

    for (var i = 0; i < rowsA; i++){
        for(var j = 0; j < colsB; j++){
            dotProducts[i][j] = DotProduct(colsA);
            for (var k = 0; k < colsA; k++){
                dotProducts[i][j].x[k] <== A[i][k];
                dotProducts[i][j].w[k] <== B[k][j];
            }
            C[i][j] <== dotProducts[i][j].out;
        }
    }
}

template EmbeddingLookup(vocab_size, d_model){
    signal input token_id;
    signal input embedding_table[vocab_size][d_model];
    signal output embedding[d_model];

    signal selectors[vocab_size];
    signal sum[d_model][vocab_size];

    for(var i = 0; i < vocab_size; i++) selectors[i] <== (token_id - i) * (token_id - i);

    for(var j = 0; j < d_model; j++){
        sum[j][0] <== embedding_table[0][j] * selectors[0];
        for(var k = 1; k < vocab_size; k++) sum[j][k] <== sum[j][k - 1] + embedding_table[k][j] * selectors[k];
        embedding[j] <== sum[j][vocab_size - 1];
    }
}


template addPositionEmbedding(seq_len, d_model){
    signal input token_embeds[seq_len][d_model];
    signal input pos_embeds[seq_len][d_model];
    signal output out[seq_len][d_model];

    for (var i = 0; i < seq_len; i++){
        for(var j = 0; j < d_model; j++){
            out[i][j] <== token_embeds[i][j] + pos_embeds[i][j];
        }
    }
}

template AttentionProjections(seq_len, d_model){
    signal input x[seq_len][d_model];
    signal input W_q[d_model][d_model];
    signal input W_k[d_model][d_model];
    signal input W_v[d_model][d_model];

    signal output Q[seq_len][d_model];
    signal output K[seq_len][d_model];
    signal output V[seq_len][d_model];

    component q_proj[seq_len];
    component k_proj[seq_len];
    component v_proj[seq_len];

    for(var i = 0; i < seq_len; i++){
        q_proj[i] = MatVecMul(d_model,d_model);
        k_proj[i] = MatVecMul(d_model,d_model);
        v_proj[i] = MatVecMul(d_model,d_model);

        for(var j = 0; j < d_model; j++){
            for(var k = 0; k < d_model; k++){
                q_proj[i].matrix[j][k] <== W_q[j][k];
                k_proj[i].matrix[j][k] <== W_k[j][k];
                v_proj[i].matrix[j][k] <== W_v[j][k];
            }
            q_proj[i].vector[j] <== x[i][j];
            k_proj[i].vector[j] <== x[i][j];
            v_proj[i].vector[j] <== x[i][j];
        }
        for(var j = 0; j < d_model; j++){
            Q[i][j] <== q_proj[i].out[j];
            K[i][j] <== k_proj[i].out[j];
            V[i][j] <== v_proj[i].out[j];
        }
    }
}

template LinearAttention(seq_len,d_model){
    signal input Q[seq_len][d_model];
    signal input K[seq_len][d_model];
    signal input V[seq_len][d_model];
    signal input W_o[d_model][d_model];

    signal output out[seq_len][d_model];
    signal sums[d_model][seq_len];
    signal aggregated[d_model];

    for(var i = 0; i < d_model; i++){
        sums[i][0] <== V[0][i];
        for(var j = 1; j < seq_len; j++) sums[i][j] <== sums[i][j - 1] + V[j][i];
        aggregated[i] <== sums[i][seq_len - 1];
    }
    
    component out_proj[seq_len];
    for(var i = 0; i < seq_len; i++){
        out_proj[i] = MatVecMul(d_model,d_model);
        for(var j = 0; j < d_model; j++){
            for(var k = 0; k < d_model; k++){
                out_proj[i].matrix[j][k] <== W_o[j][k];
            }
            out_proj[i].vector[j] <== aggregated[j];
        }
        for(var j = 0; j < d_model; j++) out[i][j] <== out_proj[i].out[j];
    }
}


template FeedFoward(seq_len,d_model,d_ff){
    signal input x[seq_len][d_model];
    signal input W_ff1[d_model][d_ff];
    signal input W_ff2[d_ff][d_model];
    signal output out[seq_len][d_model];

    component ff1[seq_len];
    component ff2[seq_len];

    signal act[seq_len][d_ff];

    for(var i = 0; i < seq_len; i++){
        ff1[i] = MatVecMul(d_ff,d_model);
        for(var j = 0; j < d_ff; j++){
            for(var k = 0; k < d_model; k++){
                ff1[i].matrix[j][k] <== W_ff1[k][j];
            }
        }
        for(var j = 0; j < d_model; j++) ff1[i].vector[j] <== x[i][j];

        for (var j = 0; j < d_ff; j++) act[i][j] <== ff1[i].out[j] * ff1[i].out[j];

        ff2[i] = MatVecMul(d_model,d_ff);
        for(var j = 0; j < d_model; j++){
            for(var k = 0; k < d_ff; k++){
                ff2[i].matrix[j][k] <== W_ff2[k][j];
            }
        }
        for (var j = 0; j < d_ff; j++) ff2[i].vector[j] <== act[i][j];

        for(var j = 0; j < d_model; j++) out[i][j] <== ff2[i].out[j];
    }
}

template SimpleLLM(vocab_size,d_model,seq_len){
    //Public Inputs
    signal input token_ids[seq_len];
    signal output predicted_token;

    //Private Inputs
    signal input token_embedding[vocab_size][d_model];
    signal input position_embedding[seq_len][d_model];
    signal input W_q[d_model][d_model];
    signal input W_k[d_model][d_model];
    signal input W_v[d_model][d_model];
    signal input W_o[d_model][d_model];
    signal input W_ff1[d_model][d_model * 2];
    signal input W_ff2[d_model * 2][d_model];
    signal input W_out[d_model][vocab_size];

    signal token_embeds[seq_len][d_model];
    component embed[seq_len];
    for(var i = 0; i < seq_len; i++){
        embed[i] = EmbeddingLookup(vocab_size,d_model);
        embed[i].token_id <== token_ids[i];
        for(var j = 0; j < vocab_size; j++){
            for(var k = 0; k < d_model; k++){
                embed[i].embedding_table[j][k] <== token_embedding[j][k];
            }
        }
        for(var j = 0; j < d_model; j++) token_embeds[i][j] <== embed[i].embedding[j];
    }

    component pos_add = addPositionEmbedding(seq_len, d_model);
    for(var i = 0; i < seq_len; i++){
        for(var j = 0; j < d_model; j++){
            pos_add.token_embeds[i][j] <== token_embeds[i][j];
            pos_add.pos_embeds[i][j] <== position_embedding[i][j];
        }
    }

    component attn_proj = AttentionProjections(seq_len,d_model);
    for(var i = 0; i < seq_len; i++){
        for(var j = 0; j < d_model; j++){
            attn_proj.x[i][j] <== pos_add.out[i][j];
        }
    }
    for(var i = 0; i < d_model; i++){
        for(var j = 0; j < d_model; j++){
            attn_proj.W_q[i][j] <== W_q[i][j];
            attn_proj.W_k[i][j] <== W_k[i][j];
            attn_proj.W_v[i][j] <== W_v[i][j];
        }
    }

    component attn = LinearAttention(seq_len,d_model);
    for(var i = 0; i < seq_len; i++){
        for(var j = 0; j < d_model; j++){
            attn.Q[i][j] <== attn_proj.Q[i][j];
            attn.K[i][j] <== attn_proj.K[i][j];
            attn.V[i][j] <== attn_proj.V[i][j];
        }
    }
    for(var i = 0; i < d_model; i++){
        for(var j = 0; j < d_model; j++){
            attn.W_o[i][j] <== W_o[i][j];
        }
    }
    
    signal post_attn[seq_len][d_model];
    for(var i = 0; i < seq_len; i++){
        for(var j = 0; j < d_model; j++){
            post_attn[i][j] <== pos_add.out[i][j] + attn.out[i][j];
        }
    }

    component ff = FeedFoward(seq_len,d_model,d_model * 2);
    for(var i = 0; i < seq_len; i++){
        for(var j = 0; j < d_model; j++){
            ff.x[i][j] <== post_attn[i][j];
        }
    }
    for(var i = 0; i < d_model; i++){
        for(var j = 0; j < d_model * 2; j++){
            ff.W_ff1[i][j] <== W_ff1[i][j];
        }
    }
    for(var i = 0; i < d_model * 2; i++){
        for(var j = 0; j < d_model; j++){
            ff.W_ff2[i][j] <== W_ff2[i][j];
        }
    }

    signal post_ff[seq_len][d_model];
    for(var i = 0; i < seq_len; i++){
        for(var j = 0; j < d_model; j++){
            post_ff[i][j] <== post_attn[i][j] + ff.out[i][j];
        }
    }

    component out_proj = MatVecMul(vocab_size,d_model);
    for(var i = 0; i < vocab_size; i++){
        for(var j = 0; j < d_model; j++){
            out_proj.matrix[i][j] <== W_out[j][i];
        }
    }
    for(var i = 0; i < d_model; i++) out_proj.vector[i] <== post_ff[seq_len - 1][i];
    predicted_token <== out_proj.out[0];
}

component main = SimpleLLM(100,32,8);