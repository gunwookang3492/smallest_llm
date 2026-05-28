package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// ---------- Core ops ----------

func layerNorm(x *mat.Dense, gamma, beta []float64, eps float64) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		row := mat.Row(nil, i, x)
		mean := 0.0
		for _, v := range row {
			mean += v
		}
		mean /= float64(c)

		variance := 0.0
		for _, v := range row {
			variance += (v - mean) * (v - mean)
		}
		std := math.Sqrt(variance / float64(c))

		for j, v := range row {
			norm := (v - mean) / (std + eps)
			out.Set(i, j, gamma[j]*norm+beta[j])
		}
	}
	return out
}

func softmaxRow(row []float64) []float64 {
	max := row[0]
	for _, v := range row {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(row))
	sum := 0.0
	for i, v := range row {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func softmax2D(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		row := mat.Row(nil, i, x)
		sm := softmaxRow(row)
		out.SetRow(i, sm)
	}
	return out
}

// ---------- Multi-head attention ----------

func multiHeadAttention(x, Wq, Wk, Wv, Wo *mat.Dense, nHeads int, mask *mat.Dense) *mat.Dense {
	seqLen, dModel := x.Dims()
	dk := dModel / nHeads

	var Q, K, V mat.Dense
	Q.Mul(x, Wq)
	K.Mul(x, Wk)
	V.Mul(x, Wv)

	// One head at a time, then concat
	heads := make([]*mat.Dense, nHeads)
	for h := 0; h < nHeads; h++ {
		Qh := sliceCols(&Q, h*dk, (h+1)*dk)
		Kh := sliceCols(&K, h*dk, (h+1)*dk)
		Vh := sliceCols(&V, h*dk, (h+1)*dk)

		var scores mat.Dense
		scores.Mul(Qh, Kh.T())
		scores.Scale(1.0/math.Sqrt(float64(dk)), &scores)
		if mask != nil {
			scores.Add(&scores, mask)
		}
		weights := softmax2D(&scores)

		var attended mat.Dense
		attended.Mul(weights, Vh)
		heads[h] = &attended
	}

	// Concatenate heads along the column axis
	concat := mat.NewDense(seqLen, dModel, nil)
	for h, head := range heads {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dk; j++ {
				concat.Set(i, h*dk+j, head.At(i, j))
			}
		}
	}

	var out mat.Dense
	out.Mul(concat, Wo)
	return &out
}

func sliceCols(m *mat.Dense, start, end int) *mat.Dense {
	r, _ := m.Dims()
	out := mat.NewDense(r, end-start, nil)
	for i := 0; i < r; i++ {
		for j := start; j < end; j++ {
			out.Set(i, j-start, m.At(i, j))
		}
	}
	return out
}

// ---------- Weights ----------

type Weights struct {
	Vocab      []string    `json:"vocab"`
	TokenEmbed [][]float64 `json:"token_embed"`
	PosEmbed   [][]float64 `json:"pos_embed"`
	Norm1W     []float64   `json:"norm1_w"`
	Norm1B     []float64   `json:"norm1_b"`
	WqT        [][]float64 `json:"wq_t"` // already transposed
	WkT        [][]float64 `json:"wk_t"`
	WvT        [][]float64 `json:"wv_t"`
	WoT        [][]float64 `json:"wo_t"`
	LnFW       []float64   `json:"ln_f_w"`
	LnFB       []float64   `json:"ln_f_b"`
	LmHead     [][]float64 `json:"lm_head"` // shape [vocab, d_model]
}

func toDense(m [][]float64) *mat.Dense {
	r := len(m)
	c := len(m[0])
	flat := make([]float64, 0, r*c)
	for _, row := range m {
		flat = append(flat, row...)
	}
	return mat.NewDense(r, c, flat)
}

func loadWeights(path string) (*Weights, error) {
	f, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var w Weights
	if err := json.Unmarshal(f, &w); err != nil {
		return nil, err
	}
	return &w, nil
}

// ---------- Predict ----------

func predictNext(prompt string, w *Weights) {
	tokens := strings.Fields(strings.ToLower(prompt))
	ids := make([]int, len(tokens))
	for i, t := range tokens {
		idx := -1
		for j, v := range w.Vocab {
			if v == t {
				idx = j
				break
			}
		}
		if idx < 0 {
			fmt.Printf("unknown token: %s\n", t)
			return
		}
		ids[i] = idx
	}

	seqLen := len(ids)
	dModel := len(w.TokenEmbed[0])

	// Embeddings + positional
	x := mat.NewDense(seqLen, dModel, nil)
	for i, id := range ids {
		for j := 0; j < dModel; j++ {
			x.Set(i, j, w.TokenEmbed[id][j]+w.PosEmbed[i][j])
		}
	}

	// Causal mask
	mask := mat.NewDense(seqLen, seqLen, nil)
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			mask.Set(i, j, -1e9)
		}
	}

	xNorm := layerNorm(x, w.Norm1W, w.Norm1B, 1e-5)
	attnOut := multiHeadAttention(
		xNorm,
		toDense(w.WqT), toDense(w.WkT),
		toDense(w.WvT), toDense(w.WoT),
		4, mask,
	)

	// Residual
	var res mat.Dense
	res.Add(x, attnOut)

	xFinal := layerNorm(&res, w.LnFW, w.LnFB, 1e-5)

	// Logits from last position
	lastRow := mat.Row(nil, seqLen-1, xFinal)
	lmHead := toDense(w.LmHead) // [vocab, d_model]
	vocabSize, _ := lmHead.Dims()
	logits := make([]float64, vocabSize)
	for v := 0; v < vocabSize; v++ {
		s := 0.0
		for j := 0; j < dModel; j++ {
			s += lastRow[j] * lmHead.At(v, j)
		}
		logits[v] = s
	}
	probs := softmaxRow(logits)

	// Top 3
	type pair struct {
		idx  int
		prob float64
	}
	pairs := make([]pair, len(probs))
	for i, p := range probs {
		pairs[i] = pair{i, p}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].prob > pairs[j].prob })

	fmt.Printf("'%s' → ", prompt)
	for i := 0; i < 3; i++ {
		fmt.Printf("%s:%.0f%% ", w.Vocab[pairs[i].idx], pairs[i].prob*100)
	}
	fmt.Println()
}

func main() {
	w, err := loadWeights("tiny_english_gpt.json")
	if err != nil {
		panic(err)
	}
	predictNext("the cat sat on the", w)
	predictNext("the dog ran to the", w)
	predictNext("the big cat sat on the big", w)
	fmt.Println("\n→ Our model predicts next words with context awareness!")
}
