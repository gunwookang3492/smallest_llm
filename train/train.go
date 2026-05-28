// Train the tiny English transformer entirely in Go, with hand-derived
// gradients (no autograd library). The exported tiny_english_gpt.json has
// the exact schema the inference code (transformer.go) expects.
//
// Architecture mirrors the inference path:
//   x = tok_emb[ids] + pos_emb[:T]
//   h = x + Attn(LN1(x))            // pre-norm, single attention sublayer
//   y = LN_f(h)
//   logits = y @ lm_head.T
//
// No FFN, no second LayerNorm — matching transformer.go exactly so the
// produced weights load cleanly.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// ---------- Config ----------

const (
	dModel    = 32
	nHeads    = 4
	dK        = dModel / nHeads
	maxSeqLen = 16
	lr        = 3e-3
	epochs    = 4000
	eps       = 1e-5
	seed      = 0
)

var corpus = []string{
	"the cat sat on the mat",
	"the cat sat on the rug",
	"the cat sat on the floor",
	"the big cat sat on the big mat",
	"the big cat sat on the big rug",
	"the small cat sat on the small mat",
	"the dog ran to the park",
	"the dog ran to the door",
	"the dog ran to the yard",
	"the big dog ran to the big park",
	"the small dog ran to the small yard",
	"the cat ran to the door",
	"the dog sat on the floor",
	"the dog sat on the mat",
	"the cat ran to the park",
}

// ---------- Tokenizer ----------

var wordRe = regexp.MustCompile(`[a-z]+`)

func buildVocab(lines []string) []string {
	set := map[string]struct{}{}
	for _, line := range lines {
		for _, w := range wordRe.FindAllString(strings.ToLower(line), -1) {
			set[w] = struct{}{}
		}
	}
	vocab := make([]string, 0, len(set))
	for w := range set {
		vocab = append(vocab, w)
	}
	sort.Strings(vocab)
	return vocab
}

func encode(text string, stoi map[string]int) []int {
	words := wordRe.FindAllString(strings.ToLower(text), -1)
	ids := make([]int, len(words))
	for i, w := range words {
		ids[i] = stoi[w]
	}
	return ids
}

// ---------- Parameters ----------
//
// We hold every learnable tensor as a *mat.Dense and keep a parallel
// gradient tensor of identical shape. Adam state (m, v) lives alongside.

type Param struct {
	W, G, M, V *mat.Dense // weight, grad, Adam m, Adam v
}

func newParam(r, c int, init func(i, j int) float64) *Param {
	w := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			w.Set(i, j, init(i, j))
		}
	}
	return &Param{
		W: w,
		G: mat.NewDense(r, c, nil),
		M: mat.NewDense(r, c, nil),
		V: mat.NewDense(r, c, nil),
	}
}

func (p *Param) zeroGrad() {
	r, c := p.G.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			p.G.Set(i, j, 0)
		}
	}
}

type Model struct {
	TokEmb, PosEmb    *Param // [V, d], [maxLen, d]
	Ln1W, Ln1B        *Param // [1, d], [1, d]    (stored as row vectors)
	Wq, Wk, Wv, Wo    *Param // each [d, d]
	LnFW, LnFB        *Param // [1, d], [1, d]
	LMHead            *Param // [V, d]
	Vocab             []string
	VocabSize, MaxLen int
}

func newModel(vocab []string, rng *rand.Rand) *Model {
	V := len(vocab)
	// Small Gaussian init for matrices; ones/zeros for LN affine params.
	gauss := func(scale float64) func(i, j int) float64 {
		return func(i, j int) float64 { return rng.NormFloat64() * scale }
	}
	ones := func(i, j int) float64 { return 1 }
	zeros := func(i, j int) float64 { return 0 }

	return &Model{
		TokEmb:    newParam(V, dModel, gauss(0.1)),
		PosEmb:    newParam(maxSeqLen, dModel, gauss(0.1)),
		Ln1W:      newParam(1, dModel, ones),
		Ln1B:      newParam(1, dModel, zeros),
		Wq:        newParam(dModel, dModel, gauss(1.0/math.Sqrt(dModel))),
		Wk:        newParam(dModel, dModel, gauss(1.0/math.Sqrt(dModel))),
		Wv:        newParam(dModel, dModel, gauss(1.0/math.Sqrt(dModel))),
		Wo:        newParam(dModel, dModel, gauss(1.0/math.Sqrt(dModel))),
		LnFW:      newParam(1, dModel, ones),
		LnFB:      newParam(1, dModel, zeros),
		LMHead:    newParam(V, dModel, gauss(0.1)),
		Vocab:     vocab,
		VocabSize: V,
		MaxLen:    maxSeqLen,
	}
}

func (m *Model) params() []*Param {
	return []*Param{
		m.TokEmb, m.PosEmb,
		m.Ln1W, m.Ln1B,
		m.Wq, m.Wk, m.Wv, m.Wo,
		m.LnFW, m.LnFB,
		m.LMHead,
	}
}

// ---------- Forward (with cached intermediates for backward) ----------
//
// We save every value the backward pass needs. The shapes use T = seq len.

type Cache struct {
	IDs []int

	X       *mat.Dense // [T, d] = tok_emb + pos_emb
	XNorm   *mat.Dense // [T, d] = LN1(X)
	Ln1Mean []float64  // per-row mean of X (for LN1 backward)
	Ln1Std  []float64  // per-row std  of X (for LN1 backward)
	Ln1Hat  *mat.Dense // [T, d] normalized (pre affine)

	Q, K, V    *mat.Dense   // [T, d]
	Qh, Kh, Vh []*mat.Dense // per-head slices [T, dK]
	Attn       []*mat.Dense // per-head softmax weights [T, T]
	Headed     []*mat.Dense // per-head attn @ V       [T, dK]
	Concat     *mat.Dense   // [T, d]

	AttnOut *mat.Dense // [T, d] concat @ Wo
	H       *mat.Dense // [T, d] X + AttnOut (residual)

	YHat            *mat.Dense // [T, d] LN_f normalized (pre affine)
	Y               *mat.Dense // [T, d] post LN_f
	LnFMean, LnFStd []float64

	Logits *mat.Dense // [T, V]
	Probs  *mat.Dense // [T, V]
}

func (m *Model) forward(ids []int) *Cache {
	T := len(ids)
	c := &Cache{IDs: ids}

	// Embedding + positional
	c.X = mat.NewDense(T, dModel, nil)
	for i, id := range ids {
		for j := 0; j < dModel; j++ {
			c.X.Set(i, j, m.TokEmb.W.At(id, j)+m.PosEmb.W.At(i, j))
		}
	}

	// LayerNorm 1
	c.XNorm, c.Ln1Hat, c.Ln1Mean, c.Ln1Std =
		layerNormFwd(c.X, m.Ln1W.W, m.Ln1B.W)

	// Attention: Q, K, V = XNorm @ {Wq, Wk, Wv}
	var Q, K, V mat.Dense
	Q.Mul(c.XNorm, m.Wq.W)
	K.Mul(c.XNorm, m.Wk.W)
	V.Mul(c.XNorm, m.Wv.W)
	c.Q, c.K, c.V = &Q, &K, &V

	c.Qh = make([]*mat.Dense, nHeads)
	c.Kh = make([]*mat.Dense, nHeads)
	c.Vh = make([]*mat.Dense, nHeads)
	c.Attn = make([]*mat.Dense, nHeads)
	c.Headed = make([]*mat.Dense, nHeads)
	c.Concat = mat.NewDense(T, dModel, nil)

	scale := 1.0 / math.Sqrt(float64(dK))
	for h := 0; h < nHeads; h++ {
		Qh := sliceCols(c.Q, h*dK, (h+1)*dK)
		Kh := sliceCols(c.K, h*dK, (h+1)*dK)
		Vh := sliceCols(c.V, h*dK, (h+1)*dK)
		c.Qh[h], c.Kh[h], c.Vh[h] = Qh, Kh, Vh

		var scores mat.Dense
		scores.Mul(Qh, Kh.T())
		scores.Scale(scale, &scores)
		// Causal mask
		for i := 0; i < T; i++ {
			for j := i + 1; j < T; j++ {
				scores.Set(i, j, math.Inf(-1))
			}
		}
		attn := softmax2D(&scores)
		c.Attn[h] = attn

		var out mat.Dense
		out.Mul(attn, Vh)
		c.Headed[h] = &out

		for i := 0; i < T; i++ {
			for j := 0; j < dK; j++ {
				c.Concat.Set(i, h*dK+j, out.At(i, j))
			}
		}
	}

	var attnOut mat.Dense
	attnOut.Mul(c.Concat, m.Wo.W)
	c.AttnOut = &attnOut

	// Residual
	c.H = mat.NewDense(T, dModel, nil)
	c.H.Add(c.X, c.AttnOut)

	// Final LayerNorm
	c.Y, c.YHat, c.LnFMean, c.LnFStd =
		layerNormFwd(c.H, m.LnFW.W, m.LnFB.W)

	// LM head: logits = Y @ lm_head.T   (since lm_head is [V, d])
	var logits mat.Dense
	logits.Mul(c.Y, m.LMHead.W.T())
	c.Logits = &logits

	// Probs for loss + backward
	c.Probs = softmax2D(&logits)
	return c
}

// ---------- Loss ----------

// Cross-entropy averaged over T positions. Returns loss and dL/dlogits.
func crossEntropyAndGrad(probs *mat.Dense, targets []int) (float64, *mat.Dense) {
	T, V := probs.Dims()
	dL := mat.NewDense(T, V, nil)
	loss := 0.0
	for i, y := range targets {
		p := probs.At(i, y)
		if p < 1e-12 {
			p = 1e-12
		}
		loss += -math.Log(p)
		for j := 0; j < V; j++ {
			dL.Set(i, j, probs.At(i, j)/float64(T))
		}
		dL.Set(i, y, dL.At(i, y)-1.0/float64(T))
	}
	return loss / float64(T), dL
}

// ---------- Backward ----------
//
// Walks the forward graph in reverse, accumulating gradients into each
// Param's .G tensor. Every op below mirrors the corresponding forward op.

func (m *Model) backward(c *Cache, dLogits *mat.Dense) {
	T, _ := c.Y.Dims()

	// --- LM head: logits = Y @ lm_head.T
	// dY        = dLogits @ lm_head
	// d(lm_head)= dLogits.T @ Y
	var dY mat.Dense
	dY.Mul(dLogits, m.LMHead.W)
	var dLM mat.Dense
	dLM.Mul(dLogits.T(), c.Y)
	addInto(m.LMHead.G, &dLM)

	// --- Final LayerNorm backward: Y = LN(H) with affine (gamma, beta)
	dH, dLnFW, dLnFB := layerNormBwd(&dY, c.H, c.YHat, c.LnFMean, c.LnFStd, m.LnFW.W)
	addInto(m.LnFW.G, dLnFW)
	addInto(m.LnFB.G, dLnFB)

	// --- Residual: H = X + AttnOut  →  dX += dH, dAttnOut = dH
	dX := mat.NewDense(T, dModel, nil)
	dX.Copy(dH)
	dAttnOut := mat.NewDense(T, dModel, nil)
	dAttnOut.Copy(dH)

	// --- Wo: AttnOut = Concat @ Wo
	var dConcat mat.Dense
	dConcat.Mul(dAttnOut, m.Wo.W.T())
	var dWo mat.Dense
	dWo.Mul(c.Concat.T(), dAttnOut)
	addInto(m.Wo.G, &dWo)

	// --- Split dConcat back into per-head dHeaded, then attention backward.
	dQ := mat.NewDense(T, dModel, nil)
	dK_ := mat.NewDense(T, dModel, nil)
	dV := mat.NewDense(T, dModel, nil)

	scale := 1.0 / math.Sqrt(float64(dK))
	for h := 0; h < nHeads; h++ {
		dHeaded := sliceCols(&dConcat, h*dK, (h+1)*dK)
		Qh, Kh, Vh := c.Qh[h], c.Kh[h], c.Vh[h]
		attn := c.Attn[h]

		// Headed = Attn @ Vh
		// dAttn = dHeaded @ Vh.T
		// dVh   = Attn.T  @ dHeaded
		var dAttn mat.Dense
		dAttn.Mul(dHeaded, Vh.T())
		var dVh mat.Dense
		dVh.Mul(attn.T(), dHeaded)

		// Softmax backward (row-wise):
		// for row i: dscores_i = (dAttn_i - sum(dAttn_i * attn_i)) * attn_i
		// The mask just made some attn entries zero; their gradient stays zero.
		dScores := mat.NewDense(T, T, nil)
		for i := 0; i < T; i++ {
			dot := 0.0
			for j := 0; j < T; j++ {
				dot += dAttn.At(i, j) * attn.At(i, j)
			}
			for j := 0; j < T; j++ {
				dScores.Set(i, j, (dAttn.At(i, j)-dot)*attn.At(i, j))
			}
		}

		// scores = (Qh @ Kh.T) * scale
		// dQh = (dScores @ Kh) * scale
		// dKh = (dScores.T @ Qh) * scale
		var dQh mat.Dense
		dQh.Mul(dScores, Kh)
		dQh.Scale(scale, &dQh)
		var dKh mat.Dense
		dKh.Mul(dScores.T(), Qh)
		dKh.Scale(scale, &dKh)

		// Place per-head grads back into full d×d Q/K/V columns.
		for i := 0; i < T; i++ {
			for j := 0; j < dK; j++ {
				dQ.Set(i, h*dK+j, dQh.At(i, j))
				dK_.Set(i, h*dK+j, dKh.At(i, j))
				dV.Set(i, h*dK+j, dVh.At(i, j))
			}
		}
	}

	// --- Q/K/V projections: Q = XNorm @ Wq, etc.
	// dXNorm += dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
	// dWq    = XNorm.T @ dQ   (similarly for Wk, Wv)
	dXNorm := mat.NewDense(T, dModel, nil)
	for _, pair := range []struct {
		dOut *mat.Dense
		W    *Param
	}{
		{dQ, m.Wq}, {dK_, m.Wk}, {dV, m.Wv},
	} {
		var contrib mat.Dense
		contrib.Mul(pair.dOut, pair.W.W.T())
		addInto(dXNorm, &contrib)

		var dW mat.Dense
		dW.Mul(c.XNorm.T(), pair.dOut)
		addInto(pair.W.G, &dW)
	}

	// --- LayerNorm 1 backward: XNorm = LN(X)
	dXFromLN, dLn1W, dLn1B := layerNormBwd(dXNorm, c.X, c.Ln1Hat, c.Ln1Mean, c.Ln1Std, m.Ln1W.W)
	addInto(m.Ln1W.G, dLn1W)
	addInto(m.Ln1B.G, dLn1B)
	addInto(dX, dXFromLN) // dX accumulates from residual AND from LN1 path

	// --- Embedding backward: X[i] = tok_emb[ids[i]] + pos_emb[i]
	for i, id := range c.IDs {
		for j := 0; j < dModel; j++ {
			g := dX.At(i, j)
			m.TokEmb.G.Set(id, j, m.TokEmb.G.At(id, j)+g)
			m.PosEmb.G.Set(i, j, m.PosEmb.G.At(i, j)+g)
		}
	}
}

// ---------- Op forward/backward helpers ----------

// LayerNorm forward.
// Returns (out, hat, mean, std). `hat` is the normalized value before the
// affine transform, kept for backward. gamma/beta are row vectors [1, d].
func layerNormFwd(x, gamma, beta *mat.Dense) (out, hat *mat.Dense, mean, std []float64) {
	r, c := x.Dims()
	out = mat.NewDense(r, c, nil)
	hat = mat.NewDense(r, c, nil)
	mean = make([]float64, r)
	std = make([]float64, r)

	for i := 0; i < r; i++ {
		// mean
		mu := 0.0
		for j := 0; j < c; j++ {
			mu += x.At(i, j)
		}
		mu /= float64(c)
		// var
		variance := 0.0
		for j := 0; j < c; j++ {
			d := x.At(i, j) - mu
			variance += d * d
		}
		s := math.Sqrt(variance/float64(c)) + eps

		mean[i] = mu
		std[i] = s
		for j := 0; j < c; j++ {
			h := (x.At(i, j) - mu) / s
			hat.Set(i, j, h)
			out.Set(i, j, gamma.At(0, j)*h+beta.At(0, j))
		}
	}
	return
}

// LayerNorm backward.
// Inputs:
//
//	dOut  [r, c]  gradient w.r.t. the LN output
//	x     [r, c]  the original input
//	hat   [r, c]  the normalized value (pre-affine) from forward
//	mean  [r]     per-row means
//	std   [r]     per-row stds (with eps already added)
//	gamma [1, c]  affine scale
//
// Returns dX, dGamma (shape [1, c]), dBeta (shape [1, c]).
//
// Derivation (per row, length n = c):
//
//	y_j = gamma_j * hat_j + beta_j
//	hat_j = (x_j - mu) / std
//	Let g_j = dOut_j * gamma_j.
//	dx_j = (1/std) * ( g_j - mean(g) - hat_j * mean(g * hat) )
func layerNormBwd(dOut, x, hat *mat.Dense, mean, std []float64, gamma *mat.Dense) (dX, dGamma, dBeta *mat.Dense) {
	r, c := x.Dims()
	dX = mat.NewDense(r, c, nil)
	dGamma = mat.NewDense(1, c, nil)
	dBeta = mat.NewDense(1, c, nil)

	for i := 0; i < r; i++ {
		// Accumulate affine grads
		for j := 0; j < c; j++ {
			dGamma.Set(0, j, dGamma.At(0, j)+dOut.At(i, j)*hat.At(i, j))
			dBeta.Set(0, j, dBeta.At(0, j)+dOut.At(i, j))
		}
		// g_j = dOut_j * gamma_j
		g := make([]float64, c)
		var gMean, gHatMean float64
		for j := 0; j < c; j++ {
			g[j] = dOut.At(i, j) * gamma.At(0, j)
			gMean += g[j]
			gHatMean += g[j] * hat.At(i, j)
		}
		gMean /= float64(c)
		gHatMean /= float64(c)

		invStd := 1.0 / std[i]
		for j := 0; j < c; j++ {
			dX.Set(i, j, invStd*(g[j]-gMean-hat.At(i, j)*gHatMean))
		}
	}
	return
}

func softmax2D(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		maxV := math.Inf(-1)
		for j := 0; j < c; j++ {
			if x.At(i, j) > maxV {
				maxV = x.At(i, j)
			}
		}
		sum := 0.0
		for j := 0; j < c; j++ {
			v := math.Exp(x.At(i, j) - maxV)
			out.Set(i, j, v)
			sum += v
		}
		for j := 0; j < c; j++ {
			out.Set(i, j, out.At(i, j)/sum)
		}
	}
	return out
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

func addInto(dst, src *mat.Dense) {
	r, c := dst.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dst.Set(i, j, dst.At(i, j)+src.At(i, j))
		}
	}
}

// ---------- Adam optimizer ----------

type Adam struct {
	Step                int
	Beta1, Beta2, EpsAd float64
}

func (a *Adam) update(p *Param) {
	a.Beta1, a.Beta2, a.EpsAd = 0.9, 0.999, 1e-8
	bc1 := 1 - math.Pow(a.Beta1, float64(a.Step))
	bc2 := 1 - math.Pow(a.Beta2, float64(a.Step))

	r, c := p.W.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			g := p.G.At(i, j)
			m := a.Beta1*p.M.At(i, j) + (1-a.Beta1)*g
			v := a.Beta2*p.V.At(i, j) + (1-a.Beta2)*g*g
			p.M.Set(i, j, m)
			p.V.Set(i, j, v)
			mHat := m / bc1
			vHat := v / bc2
			p.W.Set(i, j, p.W.At(i, j)-lr*mHat/(math.Sqrt(vHat)+a.EpsAd))
		}
	}
}

// ---------- Training loop ----------

func train() *Model {
	rng := rand.New(rand.NewSource(seed))

	vocab := buildVocab(corpus)
	stoi := map[string]int{}
	for i, w := range vocab {
		stoi[w] = i
	}
	fmt.Printf("Vocab (%d tokens): %v\n", len(vocab), vocab)

	model := newModel(vocab, rng)
	opt := &Adam{}

	// Build (input, target) examples from each sentence.
	type example struct{ in, tgt []int }
	var examples []example
	for _, line := range corpus {
		ids := encode(line, stoi)
		if len(ids) < 2 || len(ids) > maxSeqLen {
			continue
		}
		examples = append(examples, example{ids[:len(ids)-1], ids[1:]})
	}
	fmt.Printf("Training on %d sentences for %d epochs.\n", len(examples), epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		total := 0.0
		for _, ex := range examples {
			for _, p := range model.params() {
				p.zeroGrad()
			}
			cache := model.forward(ex.in)
			loss, dLogits := crossEntropyAndGrad(cache.Probs, ex.tgt)
			model.backward(cache, dLogits)

			opt.Step++
			for _, p := range model.params() {
				opt.update(p)
			}
			total += loss
		}
		if epoch == 0 || (epoch+1)%500 == 0 {
			fmt.Printf("  epoch %4d  loss=%.4f\n", epoch+1, total/float64(len(examples)))
		}
	}

	return model
}

// ---------- Prediction (sanity check) ----------

func predictNext(model *Model, stoi map[string]int, prompt string, k int) {
	ids := encode(prompt, stoi)
	cache := model.forward(ids)
	T, _ := cache.Probs.Dims()
	lastRow := mat.Row(nil, T-1, cache.Probs)

	type pair struct {
		idx  int
		prob float64
	}
	pairs := make([]pair, len(lastRow))
	for i, p := range lastRow {
		pairs[i] = pair{i, p}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].prob > pairs[j].prob })

	fmt.Printf("'%s' → ", prompt)
	for i := 0; i < k; i++ {
		fmt.Printf("%s:%.0f%% ", model.Vocab[pairs[i].idx], pairs[i].prob*100)
	}
	fmt.Println()
}

// ---------- Export ----------

func denseToSlice(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		out[i] = make([]float64, c)
		for j := 0; j < c; j++ {
			out[i][j] = m.At(i, j)
		}
	}
	return out
}

func rowVec(m *mat.Dense) []float64 {
	_, c := m.Dims()
	out := make([]float64, c)
	for j := 0; j < c; j++ {
		out[j] = m.At(0, j)
	}
	return out
}

type weightsJSON struct {
	Vocab      []string    `json:"vocab"`
	TokenEmbed [][]float64 `json:"token_embed"`
	PosEmbed   [][]float64 `json:"pos_embed"`
	Norm1W     []float64   `json:"norm1_w"`
	Norm1B     []float64   `json:"norm1_b"`
	WqT        [][]float64 `json:"wq_t"`
	WkT        [][]float64 `json:"wk_t"`
	WvT        [][]float64 `json:"wv_t"`
	WoT        [][]float64 `json:"wo_t"`
	LnFW       []float64   `json:"ln_f_w"`
	LnFB       []float64   `json:"ln_f_b"`
	LmHead     [][]float64 `json:"lm_head"`
}

func exportWeights(model *Model, path string) error {
	// Our Wq/Wk/Wv/Wo are already stored in the [d, d] orientation that
	// the inference code multiplies on the right (x @ Wq). No transpose
	// needed — the `_t` field name is preserved from the original schema.
	data := weightsJSON{
		Vocab:      model.Vocab,
		TokenEmbed: denseToSlice(model.TokEmb.W),
		PosEmbed:   denseToSlice(model.PosEmb.W),
		Norm1W:     rowVec(model.Ln1W.W),
		Norm1B:     rowVec(model.Ln1B.W),
		WqT:        denseToSlice(model.Wq.W),
		WkT:        denseToSlice(model.Wk.W),
		WvT:        denseToSlice(model.Wv.W),
		WoT:        denseToSlice(model.Wo.W),
		LnFW:       rowVec(model.LnFW.W),
		LnFB:       rowVec(model.LnFB.W),
		LmHead:     denseToSlice(model.LMHead.W),
	}
	bytes, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return os.WriteFile(path, bytes, 0644)
}

// ---------- Main ----------

func main() {
	model := train()

	stoi := map[string]int{}
	for i, w := range model.Vocab {
		stoi[w] = i
	}
	fmt.Println("\nGo-side predictions (should match the Python output):")
	predictNext(model, stoi, "the cat sat on the", 3)
	predictNext(model, stoi, "the dog ran to the", 3)
	predictNext(model, stoi, "the big cat sat on the big", 3)

	if err := exportWeights(model, "tiny_english_gpt.json"); err != nil {
		panic(err)
	}
	fmt.Println("Wrote tiny_english_gpt.json")
}
