// Train a small char-level transformer entirely in Go, with hand-derived
// gradients (no autograd library). This is the "scaled up" version of the
// original single-block model: it now has a proper pre-norm transformer block
// repeated n_layers times, each with BOTH an attention sublayer and a
// GELU feed-forward (FFN) sublayer.
//
// Architecture (per block, pre-norm):
//   x = x + Attn(LN1(x))
//   x = x + FFN(LN2(x))      // FFN: Linear(d, d_ff) -> GELU -> Linear(d_ff, d)
// then once at the end:
//   y      = LN_f(x)
//   logits = y @ lm_head.T
//
// Tokenizer is character-level: each vocab entry is a single character, plus
// "\n" used as the end-of-story / end-of-sentence token. The exported
// tiny_english_gpt.json carries a `config` block and a `blocks` array so the
// inference code can reconstruct an arbitrary-depth model.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// ---------- Config ----------
//
// Tunable via environment variables so we can experiment without recompiling.
// Defaults are sized for a multi-MB char-level run on CPU.

const (
	eps  = 1e-5
	seed = 0
)

var (
	dModel    = envInt("D_MODEL", 64)
	nHeads    = envInt("N_HEADS", 4)
	nLayers   = envInt("N_LAYERS", 3)
	dFF       = envInt("D_FF", 256)
	maxSeqLen = envInt("MAX_SEQ", 128)
	lr        = envFloat("LR", 5e-4)
	batchSize = envInt("BATCH", 16)
	maxSteps  = envInt("STEPS", 3000)
	logEvery  = envInt("LOG_EVERY", 50)
	ckptEvery = envInt("CKPT_EVERY", 200)
	dataPath  = envStr("DATA", "tinystories_raw.txt")
	outPath   = envStr("OUT", "tiny_english_gpt.json")
	dK        int // = dModel / nHeads, set in main
)

func envStr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(k string, def float64) float64 {
	if v := os.Getenv(k); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// Fallback toy corpus, used only when the data file is missing.
var toyCorpus = []string{
	"the cat sat on the mat",
	"the dog ran to the park",
	"the big cat sat on the big rug",
}

// loadCorpus reads the dataset file, splits on the TinyStories story separator,
// lowercases, and collapses all whitespace within a story to single spaces.
func loadCorpus(path string) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("(no %s — falling back to toy corpus: %v)\n", path, err)
		return toyCorpus
	}
	text := strings.ToLower(string(b))
	raw := strings.Split(text, "<|endoftext|>")
	out := make([]string, 0, len(raw))
	for _, s := range raw {
		s = strings.Join(strings.Fields(s), " ")
		if len(s) >= 16 { // drop fragments / the partial first & last stories
			out = append(out, s)
		}
	}
	return out
}

// ---------- Char-level tokenizer ----------

const eosTok = "\n" // end-of-story marker

func buildVocab(lines []string) []string {
	set := map[string]struct{}{eosTok: {}}
	for _, line := range lines {
		for _, r := range strings.ToLower(line) {
			set[string(r)] = struct{}{}
		}
	}
	vocab := make([]string, 0, len(set))
	for c := range set {
		vocab = append(vocab, c)
	}
	sort.Strings(vocab)
	return vocab
}

// encode turns text into char ids, skipping any rune not in the vocab.
func encode(text string, stoi map[string]int) []int {
	ids := make([]int, 0, len(text))
	for _, r := range strings.ToLower(text) {
		if id, ok := stoi[string(r)]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// ---------- Parameters ----------

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

// Block holds the learnable tensors of one pre-norm transformer block.
type Block struct {
	Ln1W, Ln1B     *Param // [1, d]
	Wq, Wk, Wv, Wo *Param // [d, d]
	Ln2W, Ln2B     *Param // [1, d]
	Fc1W           *Param // [d, d_ff]
	Fc1B           *Param // [1, d_ff]
	Fc2W           *Param // [d_ff, d]
	Fc2B           *Param // [1, d]
}

type Model struct {
	TokEmb, PosEmb    *Param // [V, d], [maxLen, d]
	Blocks            []*Block
	LnFW, LnFB        *Param // [1, d]
	LMHead            *Param // [V, d]
	Vocab             []string
	VocabSize, MaxLen int
}

func newModel(vocab []string, rng *rand.Rand) *Model {
	V := len(vocab)
	gauss := func(scale float64) func(i, j int) float64 {
		return func(i, j int) float64 { return rng.NormFloat64() * scale }
	}
	ones := func(i, j int) float64 { return 1 }
	zeros := func(i, j int) float64 { return 0 }

	newBlock := func() *Block {
		return &Block{
			Ln1W: newParam(1, dModel, ones),
			Ln1B: newParam(1, dModel, zeros),
			Wq:   newParam(dModel, dModel, gauss(1.0/math.Sqrt(float64(dModel)))),
			Wk:   newParam(dModel, dModel, gauss(1.0/math.Sqrt(float64(dModel)))),
			Wv:   newParam(dModel, dModel, gauss(1.0/math.Sqrt(float64(dModel)))),
			Wo:   newParam(dModel, dModel, gauss(1.0/math.Sqrt(float64(dModel)))),
			Ln2W: newParam(1, dModel, ones),
			Ln2B: newParam(1, dModel, zeros),
			Fc1W: newParam(dModel, dFF, gauss(1.0/math.Sqrt(float64(dModel)))),
			Fc1B: newParam(1, dFF, zeros),
			Fc2W: newParam(dFF, dModel, gauss(1.0/math.Sqrt(float64(dFF)))),
			Fc2B: newParam(1, dModel, zeros),
		}
	}

	blocks := make([]*Block, nLayers)
	for i := range blocks {
		blocks[i] = newBlock()
	}

	return &Model{
		TokEmb:    newParam(V, dModel, gauss(0.1)),
		PosEmb:    newParam(maxSeqLen, dModel, gauss(0.1)),
		Blocks:    blocks,
		LnFW:      newParam(1, dModel, ones),
		LnFB:      newParam(1, dModel, zeros),
		LMHead:    newParam(V, dModel, gauss(0.1)),
		Vocab:     vocab,
		VocabSize: V,
		MaxLen:    maxSeqLen,
	}
}

func (b *Block) params() []*Param {
	return []*Param{
		b.Ln1W, b.Ln1B,
		b.Wq, b.Wk, b.Wv, b.Wo,
		b.Ln2W, b.Ln2B,
		b.Fc1W, b.Fc1B, b.Fc2W, b.Fc2B,
	}
}

func (m *Model) params() []*Param {
	ps := []*Param{m.TokEmb, m.PosEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.params()...)
	}
	ps = append(ps, m.LnFW, m.LnFB, m.LMHead)
	return ps
}

// ---------- Forward (with cached intermediates for backward) ----------

// BlockCache stores every value the block's backward pass needs. T = seq len.
type BlockCache struct {
	XIn *mat.Dense // [T, d] block input

	N1     *mat.Dense // LN1 output (post-affine)
	N1Hat  *mat.Dense // LN1 normalized (pre-affine)
	N1Mean []float64
	N1Std  []float64

	Q, K, V    *mat.Dense
	Qh, Kh, Vh []*mat.Dense
	Attn       []*mat.Dense // softmax weights [T, T] per head
	Headed     []*mat.Dense // attn @ V per head
	Concat     *mat.Dense
	AttnOut    *mat.Dense // concat @ Wo
	XMid       *mat.Dense // XIn + AttnOut (residual 1)

	N2     *mat.Dense // LN2 output (post-affine)
	N2Hat  *mat.Dense
	N2Mean []float64
	N2Std  []float64

	Z1     *mat.Dense // N2 @ Fc1W + b1  (pre-GELU)
	G      *mat.Dense // GELU(Z1)
	GDeriv *mat.Dense // GELU'(Z1)
	FF     *mat.Dense // G @ Fc2W + b2
	XOut   *mat.Dense // XMid + FF (residual 2)
}

type Cache struct {
	IDs    []int
	X      *mat.Dense // tok_emb + pos_emb
	Blocks []*BlockCache

	YHat            *mat.Dense // LN_f normalized (pre-affine)
	Y               *mat.Dense // LN_f output
	LnFMean, LnFStd []float64

	Logits *mat.Dense
	Probs  *mat.Dense
}

func (m *Model) forward(ids []int) *Cache {
	T := len(ids)
	c := &Cache{IDs: ids, Blocks: make([]*BlockCache, len(m.Blocks))}

	// Embedding + positional
	c.X = mat.NewDense(T, dModel, nil)
	for i, id := range ids {
		for j := 0; j < dModel; j++ {
			c.X.Set(i, j, m.TokEmb.W.At(id, j)+m.PosEmb.W.At(i, j))
		}
	}

	x := c.X
	scale := 1.0 / math.Sqrt(float64(dK))
	for bi, blk := range m.Blocks {
		bc := &BlockCache{XIn: x}

		// --- LayerNorm 1
		bc.N1, bc.N1Hat, bc.N1Mean, bc.N1Std = layerNormFwd(x, blk.Ln1W.W, blk.Ln1B.W)

		// --- Attention: Q,K,V = N1 @ {Wq,Wk,Wv}
		var Q, K, V mat.Dense
		Q.Mul(bc.N1, blk.Wq.W)
		K.Mul(bc.N1, blk.Wk.W)
		V.Mul(bc.N1, blk.Wv.W)
		bc.Q, bc.K, bc.V = &Q, &K, &V

		bc.Qh = make([]*mat.Dense, nHeads)
		bc.Kh = make([]*mat.Dense, nHeads)
		bc.Vh = make([]*mat.Dense, nHeads)
		bc.Attn = make([]*mat.Dense, nHeads)
		bc.Headed = make([]*mat.Dense, nHeads)
		bc.Concat = mat.NewDense(T, dModel, nil)

		for h := 0; h < nHeads; h++ {
			Qh := sliceCols(bc.Q, h*dK, (h+1)*dK)
			Kh := sliceCols(bc.K, h*dK, (h+1)*dK)
			Vh := sliceCols(bc.V, h*dK, (h+1)*dK)
			bc.Qh[h], bc.Kh[h], bc.Vh[h] = Qh, Kh, Vh

			var scores mat.Dense
			scores.Mul(Qh, Kh.T())
			scores.Scale(scale, &scores)
			for i := 0; i < T; i++ {
				for j := i + 1; j < T; j++ {
					scores.Set(i, j, math.Inf(-1))
				}
			}
			attn := softmax2D(&scores)
			bc.Attn[h] = attn

			var out mat.Dense
			out.Mul(attn, Vh)
			bc.Headed[h] = &out

			for i := 0; i < T; i++ {
				for j := 0; j < dK; j++ {
					bc.Concat.Set(i, h*dK+j, out.At(i, j))
				}
			}
		}

		var attnOut mat.Dense
		attnOut.Mul(bc.Concat, blk.Wo.W)
		bc.AttnOut = &attnOut

		// --- Residual 1
		bc.XMid = mat.NewDense(T, dModel, nil)
		bc.XMid.Add(x, bc.AttnOut)

		// --- LayerNorm 2
		bc.N2, bc.N2Hat, bc.N2Mean, bc.N2Std = layerNormFwd(bc.XMid, blk.Ln2W.W, blk.Ln2B.W)

		// --- FFN: Z1 = N2 @ Fc1W + b1 ; G = GELU(Z1) ; FF = G @ Fc2W + b2
		var z1 mat.Dense
		z1.Mul(bc.N2, blk.Fc1W.W)
		addRowVec(&z1, blk.Fc1B.W)
		bc.Z1 = &z1
		bc.G, bc.GDeriv = geluFwd(&z1)

		var ff mat.Dense
		ff.Mul(bc.G, blk.Fc2W.W)
		addRowVec(&ff, blk.Fc2B.W)
		bc.FF = &ff

		// --- Residual 2
		bc.XOut = mat.NewDense(T, dModel, nil)
		bc.XOut.Add(bc.XMid, bc.FF)

		c.Blocks[bi] = bc
		x = bc.XOut
	}

	// Final LayerNorm + LM head
	c.Y, c.YHat, c.LnFMean, c.LnFStd = layerNormFwd(x, m.LnFW.W, m.LnFB.W)
	var logits mat.Dense
	logits.Mul(c.Y, m.LMHead.W.T())
	c.Logits = &logits
	c.Probs = softmax2D(&logits)
	return c
}

// ---------- Loss ----------

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

// GradSet is a private, per-worker set of gradient tensors (one per Param).
// Concurrent backward passes each write into their own GradSet, which the
// training loop then reduces into the shared param grads — no locking needed.
type GradSet struct{ g map[*Param]*mat.Dense }

func newGradSet(m *Model) *GradSet {
	gs := &GradSet{g: make(map[*Param]*mat.Dense)}
	for _, p := range m.params() {
		r, c := p.W.Dims()
		gs.g[p] = mat.NewDense(r, c, nil)
	}
	return gs
}

func (gs *GradSet) add(p *Param, d *mat.Dense) { addInto(gs.g[p], d) }

func (m *Model) backward(c *Cache, dLogits *mat.Dense, gs *GradSet) {
	T, _ := c.Y.Dims()

	// --- LM head: logits = Y @ lm_head.T
	var dY mat.Dense
	dY.Mul(dLogits, m.LMHead.W)
	var dLM mat.Dense
	dLM.Mul(dLogits.T(), c.Y)
	gs.add(m.LMHead, &dLM)

	// --- Final LayerNorm: Y = LN(lastXOut)
	lastX := c.Blocks[len(c.Blocks)-1].XOut
	dx, dLnFW, dLnFB := layerNormBwd(&dY, lastX, c.YHat, c.LnFMean, c.LnFStd, m.LnFW.W)
	gs.add(m.LnFW, dLnFW)
	gs.add(m.LnFB, dLnFB)

	// --- Blocks in reverse
	for bi := len(m.Blocks) - 1; bi >= 0; bi-- {
		dx = m.blockBackward(m.Blocks[bi], c.Blocks[bi], dx, T, gs)
	}

	// --- Embedding: X[i] = tok_emb[ids[i]] + pos_emb[i]
	tg := gs.g[m.TokEmb]
	pg := gs.g[m.PosEmb]
	for i, id := range c.IDs {
		for j := 0; j < dModel; j++ {
			g := dx.At(i, j)
			tg.Set(id, j, tg.At(id, j)+g)
			pg.Set(i, j, pg.At(i, j)+g)
		}
	}
}

// blockBackward consumes dXOut (grad w.r.t. this block's output) and returns
// dXIn (grad w.r.t. its input), accumulating into the worker's GradSet.
func (m *Model) blockBackward(blk *Block, bc *BlockCache, dXOut *mat.Dense, T int, gs *GradSet) *mat.Dense {
	// --- Residual 2: XOut = XMid + FF  →  dXMid += dXOut, dFF = dXOut
	dXMid := mat.NewDense(T, dModel, nil)
	dXMid.Copy(dXOut)
	dff := dXOut // read-only alias

	// --- FFN backward.
	// FF = G @ Fc2W + b2
	var dFc2W mat.Dense
	dFc2W.Mul(bc.G.T(), dff)
	gs.add(blk.Fc2W, &dFc2W)
	gs.add(blk.Fc2B, colSum(dff))
	var dG mat.Dense
	dG.Mul(dff, blk.Fc2W.W.T())

	// Z1 -> G via GELU: dZ1 = dG * GELU'(Z1)
	dZ1 := mat.NewDense(T, dFF, nil)
	for i := 0; i < T; i++ {
		for j := 0; j < dFF; j++ {
			dZ1.Set(i, j, dG.At(i, j)*bc.GDeriv.At(i, j))
		}
	}
	// Z1 = N2 @ Fc1W + b1
	var dFc1W mat.Dense
	dFc1W.Mul(bc.N2.T(), dZ1)
	gs.add(blk.Fc1W, &dFc1W)
	gs.add(blk.Fc1B, colSum(dZ1))
	var dN2 mat.Dense
	dN2.Mul(dZ1, blk.Fc1W.W.T())

	// --- LayerNorm 2 backward: N2 = LN(XMid)
	dxMidFromLN, dLn2W, dLn2B := layerNormBwd(&dN2, bc.XMid, bc.N2Hat, bc.N2Mean, bc.N2Std, blk.Ln2W.W)
	gs.add(blk.Ln2W, dLn2W)
	gs.add(blk.Ln2B, dLn2B)
	addInto(dXMid, dxMidFromLN)

	// --- Residual 1: XMid = XIn + AttnOut  →  dXIn += dXMid, dAttnOut = dXMid
	dXIn := mat.NewDense(T, dModel, nil)
	dXIn.Copy(dXMid)
	dAttnOut := mat.NewDense(T, dModel, nil)
	dAttnOut.Copy(dXMid)

	// --- Wo: AttnOut = Concat @ Wo
	var dConcat mat.Dense
	dConcat.Mul(dAttnOut, blk.Wo.W.T())
	var dWo mat.Dense
	dWo.Mul(bc.Concat.T(), dAttnOut)
	gs.add(blk.Wo, &dWo)

	// --- Per-head attention backward
	dQ := mat.NewDense(T, dModel, nil)
	dKk := mat.NewDense(T, dModel, nil)
	dV := mat.NewDense(T, dModel, nil)
	scale := 1.0 / math.Sqrt(float64(dK))
	for h := 0; h < nHeads; h++ {
		dHeaded := sliceCols(&dConcat, h*dK, (h+1)*dK)
		Qh, Kh, Vh := bc.Qh[h], bc.Kh[h], bc.Vh[h]
		attn := bc.Attn[h]

		// Headed = Attn @ Vh
		var dAttn mat.Dense
		dAttn.Mul(dHeaded, Vh.T())
		var dVh mat.Dense
		dVh.Mul(attn.T(), dHeaded)

		// Softmax backward (row-wise)
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
		var dQh mat.Dense
		dQh.Mul(dScores, Kh)
		dQh.Scale(scale, &dQh)
		var dKh mat.Dense
		dKh.Mul(dScores.T(), Qh)
		dKh.Scale(scale, &dKh)

		for i := 0; i < T; i++ {
			for j := 0; j < dK; j++ {
				dQ.Set(i, h*dK+j, dQh.At(i, j))
				dKk.Set(i, h*dK+j, dKh.At(i, j))
				dV.Set(i, h*dK+j, dVh.At(i, j))
			}
		}
	}

	// --- Q/K/V projections: Q = N1 @ Wq, etc.
	dN1 := mat.NewDense(T, dModel, nil)
	for _, pair := range []struct {
		dOut *mat.Dense
		W    *Param
	}{
		{dQ, blk.Wq}, {dKk, blk.Wk}, {dV, blk.Wv},
	} {
		var contrib mat.Dense
		contrib.Mul(pair.dOut, pair.W.W.T())
		addInto(dN1, &contrib)

		var dW mat.Dense
		dW.Mul(bc.N1.T(), pair.dOut)
		gs.add(pair.W, &dW)
	}

	// --- LayerNorm 1 backward: N1 = LN(XIn)
	dxInFromLN, dLn1W, dLn1B := layerNormBwd(dN1, bc.XIn, bc.N1Hat, bc.N1Mean, bc.N1Std, blk.Ln1W.W)
	gs.add(blk.Ln1W, dLn1W)
	gs.add(blk.Ln1B, dLn1B)
	addInto(dXIn, dxInFromLN)

	return dXIn
}

// ---------- Op forward/backward helpers ----------

func layerNormFwd(x, gamma, beta *mat.Dense) (out, hat *mat.Dense, mean, std []float64) {
	r, c := x.Dims()
	out = mat.NewDense(r, c, nil)
	hat = mat.NewDense(r, c, nil)
	mean = make([]float64, r)
	std = make([]float64, r)

	for i := 0; i < r; i++ {
		mu := 0.0
		for j := 0; j < c; j++ {
			mu += x.At(i, j)
		}
		mu /= float64(c)
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

func layerNormBwd(dOut, x, hat *mat.Dense, mean, std []float64, gamma *mat.Dense) (dX, dGamma, dBeta *mat.Dense) {
	r, c := x.Dims()
	dX = mat.NewDense(r, c, nil)
	dGamma = mat.NewDense(1, c, nil)
	dBeta = mat.NewDense(1, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dGamma.Set(0, j, dGamma.At(0, j)+dOut.At(i, j)*hat.At(i, j))
			dBeta.Set(0, j, dBeta.At(0, j)+dOut.At(i, j))
		}
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

// geluFwd returns GELU(z) and GELU'(z) elementwise (exact erf form).
func geluFwd(z *mat.Dense) (out, deriv *mat.Dense) {
	const invSqrt2 = 0.7071067811865476   // 1/sqrt(2)
	const invSqrt2pi = 0.3989422804014327 // 1/sqrt(2*pi)
	r, c := z.Dims()
	out = mat.NewDense(r, c, nil)
	deriv = mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			x := z.At(i, j)
			cdf := 0.5 * (1 + math.Erf(x*invSqrt2))
			pdf := invSqrt2pi * math.Exp(-0.5*x*x)
			out.Set(i, j, x*cdf)
			deriv.Set(i, j, cdf+x*pdf)
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

// addRowVec adds row vector b [1,c] to every row of m [r,c] in place.
func addRowVec(m, b *mat.Dense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, m.At(i, j)+b.At(0, j))
		}
	}
}

// colSum sums m [r,c] over rows, returning a [1,c] row vector (bias gradient).
func colSum(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(1, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Set(0, j, out.At(0, j)+m.At(i, j))
		}
	}
	return out
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
			mv := a.Beta1*p.M.At(i, j) + (1-a.Beta1)*g
			vv := a.Beta2*p.V.At(i, j) + (1-a.Beta2)*g*g
			p.M.Set(i, j, mv)
			p.V.Set(i, j, vv)
			mHat := mv / bc1
			vHat := vv / bc2
			p.W.Set(i, j, p.W.At(i, j)-lr*mHat/(math.Sqrt(vHat)+a.EpsAd))
		}
	}
}

// ---------- Training loop ----------

type example struct{ in, tgt []int }

// buildExamples concatenates every story (each terminated by EOS) into one
// token stream, then cuts it into non-overlapping windows of maxSeqLen+1 so
// each window yields (input, shifted-target).
func buildExamples(stories []string, stoi map[string]int) []example {
	var stream []int
	for _, s := range stories {
		stream = append(stream, encode(s+eosTok, stoi)...)
	}
	win := maxSeqLen + 1
	var ex []example
	for i := 0; i+win <= len(stream); i += maxSeqLen {
		w := stream[i : i+win]
		in := append([]int(nil), w[:maxSeqLen]...)
		tgt := append([]int(nil), w[1:]...)
		ex = append(ex, example{in, tgt})
	}
	return ex
}

func scaleGrad(p *Param, s float64) {
	r, c := p.G.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			p.G.Set(i, j, p.G.At(i, j)*s)
		}
	}
}

func train() *Model {
	rng := rand.New(rand.NewSource(seed))

	stories := loadCorpus(dataPath)
	vocab := buildVocab(stories)
	stoi := map[string]int{}
	for i, w := range vocab {
		stoi[w] = i
	}
	fmt.Printf("Vocab (%d chars): %q\n", len(vocab), strings.Join(vocab, ""))

	model := newModel(vocab, rng)
	opt := &Adam{}

	examples := buildExamples(stories, stoi)
	if len(examples) < batchSize {
		panic("not enough data for one batch")
	}
	fmt.Printf("%d stories → %d windows. Config: d=%d heads=%d layers=%d d_ff=%d seq=%d batch=%d steps=%d lr=%g\n",
		len(stories), len(examples), dModel, nHeads, nLayers, dFF, maxSeqLen, batchSize, maxSteps, lr)

	invB := 1.0 / float64(batchSize)
	workers := runtime.NumCPU()
	params := model.params()
	fmt.Printf("Parallel batch over %d workers.\n", workers)

	step := 0
	for step < maxSteps {
		rng.Shuffle(len(examples), func(i, j int) { examples[i], examples[j] = examples[j], examples[i] })
		for bStart := 0; bStart+batchSize <= len(examples) && step < maxSteps; bStart += batchSize {
			// Run the batch's examples concurrently; each worker accumulates
			// into its own GradSet (no shared writes during backward).
			grads := make([]*GradSet, batchSize)
			losses := make([]float64, batchSize)
			var wg sync.WaitGroup
			sem := make(chan struct{}, workers)
			for k := 0; k < batchSize; k++ {
				wg.Add(1)
				go func(k int) {
					defer wg.Done()
					sem <- struct{}{}
					defer func() { <-sem }()
					ex := examples[bStart+k]
					gs := newGradSet(model)
					cache := model.forward(ex.in)
					loss, dLogits := crossEntropyAndGrad(cache.Probs, ex.tgt)
					model.backward(cache, dLogits, gs)
					grads[k] = gs
					losses[k] = loss
				}(k)
			}
			wg.Wait()

			// Reduce per-worker grads into the shared param grads.
			for _, p := range params {
				p.zeroGrad()
			}
			batchLoss := 0.0
			for k := 0; k < batchSize; k++ {
				batchLoss += losses[k]
				for _, p := range params {
					addInto(p.G, grads[k].g[p])
				}
			}
			for _, p := range params {
				scaleGrad(p, invB)
			}
			opt.Step++
			for _, p := range params {
				opt.update(p)
			}
			step++

			if step == 1 || step%logEvery == 0 {
				fmt.Printf("  step %5d/%d  loss=%.4f\n", step, maxSteps, batchLoss*invB)
			}
			if step%ckptEvery == 0 {
				if err := exportWeights(model, outPath); err != nil {
					fmt.Println("checkpoint error:", err)
				} else {
					sample := generate(model, stoi, "once upon a time", 120)
					fmt.Printf("  [ckpt @ %d] %q\n", step, sample)
				}
			}
		}
	}
	return model
}

// ---------- Prediction (sanity check) ----------

// generate greedily continues prompt char-by-char until EOS or maxNew chars.
func generate(model *Model, stoi map[string]int, prompt string, maxNew int) string {
	ids := encode(prompt, stoi)
	for n := 0; n < maxNew; n++ {
		ctx := ids
		if len(ctx) > maxSeqLen {
			ctx = ctx[len(ctx)-maxSeqLen:]
		}
		cache := model.forward(ctx)
		T, _ := cache.Probs.Dims()
		row := mat.Row(nil, T-1, cache.Probs)
		best := 0
		for i, p := range row {
			if p > row[best] {
				best = i
			}
		}
		if model.Vocab[best] == eosTok {
			break
		}
		ids = append(ids, best)
	}
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(model.Vocab[id])
	}
	return sb.String()
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

type configJSON struct {
	DModel    int `json:"d_model"`
	NHeads    int `json:"n_heads"`
	NLayers   int `json:"n_layers"`
	DFF       int `json:"d_ff"`
	MaxSeqLen int `json:"max_seq_len"`
}

type blockJSON struct {
	Ln1W []float64   `json:"ln1_w"`
	Ln1B []float64   `json:"ln1_b"`
	Wq   [][]float64 `json:"wq"`
	Wk   [][]float64 `json:"wk"`
	Wv   [][]float64 `json:"wv"`
	Wo   [][]float64 `json:"wo"`
	Ln2W []float64   `json:"ln2_w"`
	Ln2B []float64   `json:"ln2_b"`
	Fc1W [][]float64 `json:"fc1_w"`
	Fc1B []float64   `json:"fc1_b"`
	Fc2W [][]float64 `json:"fc2_w"`
	Fc2B []float64   `json:"fc2_b"`
}

type weightsJSON struct {
	Config     configJSON  `json:"config"`
	Vocab      []string    `json:"vocab"`
	TokenEmbed [][]float64 `json:"token_embed"`
	PosEmbed   [][]float64 `json:"pos_embed"`
	Blocks     []blockJSON `json:"blocks"`
	LnFW       []float64   `json:"ln_f_w"`
	LnFB       []float64   `json:"ln_f_b"`
	LmHead     [][]float64 `json:"lm_head"`
}

func exportWeights(model *Model, path string) error {
	blocks := make([]blockJSON, len(model.Blocks))
	for i, b := range model.Blocks {
		blocks[i] = blockJSON{
			Ln1W: rowVec(b.Ln1W.W), Ln1B: rowVec(b.Ln1B.W),
			Wq: denseToSlice(b.Wq.W), Wk: denseToSlice(b.Wk.W),
			Wv: denseToSlice(b.Wv.W), Wo: denseToSlice(b.Wo.W),
			Ln2W: rowVec(b.Ln2W.W), Ln2B: rowVec(b.Ln2B.W),
			Fc1W: denseToSlice(b.Fc1W.W), Fc1B: rowVec(b.Fc1B.W),
			Fc2W: denseToSlice(b.Fc2W.W), Fc2B: rowVec(b.Fc2B.W),
		}
	}
	data := weightsJSON{
		Config: configJSON{
			DModel: dModel, NHeads: nHeads, NLayers: nLayers,
			DFF: dFF, MaxSeqLen: maxSeqLen,
		},
		Vocab:      model.Vocab,
		TokenEmbed: denseToSlice(model.TokEmb.W),
		PosEmbed:   denseToSlice(model.PosEmb.W),
		Blocks:     blocks,
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
	dK = dModel / nHeads
	if dModel%nHeads != 0 {
		panic("d_model must be divisible by n_heads")
	}

	model := train()

	stoi := map[string]int{}
	for i, w := range model.Vocab {
		stoi[w] = i
	}
	fmt.Println("\nGreedy generations (char-level):")
	for _, p := range []string{"once upon a time", "the little", "one day"} {
		fmt.Printf("%q → %q\n", p, generate(model, stoi, p, 150))
	}

	if err := exportWeights(model, outPath); err != nil {
		panic(err)
	}
	fmt.Printf("Wrote %s\n", outPath)
}
