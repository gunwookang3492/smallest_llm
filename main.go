package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

const eosTok = "\n"

// Sampling knobs, tunable via env (GEN_TEMP, GEN_TOPK). Lower temp = steadier/
// more repetitive; higher = more varied/risky. (Avoid the name TEMP — it's a
// reserved Windows env var for the temp directory.)
var (
	genTemp = envFloat("GEN_TEMP", 0.8)
	genTopK = envInt("GEN_TOPK", 10)
)

func envFloat(k string, def float64) float64 {
	if v := os.Getenv(k); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
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
		std := math.Sqrt(variance/float64(c)) + eps

		for j, v := range row {
			out.Set(i, j, gamma[j]*((v-mean)/std)+beta[j])
		}
	}
	return out
}

func softmaxRow(row []float64) []float64 {
	maxV := math.Inf(-1)
	for _, v := range row {
		if v > maxV {
			maxV = v
		}
	}
	out := make([]float64, len(row))
	sum := 0.0
	for i, v := range row {
		out[i] = math.Exp(v - maxV)
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
		out.SetRow(i, softmaxRow(mat.Row(nil, i, x)))
	}
	return out
}

// gelu applies the exact (erf-form) GELU elementwise.
func gelu(z *mat.Dense) *mat.Dense {
	const invSqrt2 = 0.7071067811865476
	r, c := z.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			x := z.At(i, j)
			out.Set(i, j, x*0.5*(1+math.Erf(x*invSqrt2)))
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

// addRowVec adds row vector b to every row of m in place.
func addRowVec(m *mat.Dense, b []float64) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, m.At(i, j)+b[j])
		}
	}
}

// ---------- Multi-head attention (causal) ----------

func multiHeadAttention(x, Wq, Wk, Wv, Wo *mat.Dense, nHeads int) *mat.Dense {
	seqLen, dModel := x.Dims()
	dk := dModel / nHeads

	var Q, K, V mat.Dense
	Q.Mul(x, Wq)
	K.Mul(x, Wk)
	V.Mul(x, Wv)

	concat := mat.NewDense(seqLen, dModel, nil)
	scale := 1.0 / math.Sqrt(float64(dk))
	for h := 0; h < nHeads; h++ {
		Qh := sliceCols(&Q, h*dk, (h+1)*dk)
		Kh := sliceCols(&K, h*dk, (h+1)*dk)
		Vh := sliceCols(&V, h*dk, (h+1)*dk)

		var scores mat.Dense
		scores.Mul(Qh, Kh.T())
		scores.Scale(scale, &scores)
		for i := 0; i < seqLen; i++ {
			for j := i + 1; j < seqLen; j++ {
				scores.Set(i, j, math.Inf(-1))
			}
		}
		weights := softmax2D(&scores)

		var attended mat.Dense
		attended.Mul(weights, Vh)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dk; j++ {
				concat.Set(i, h*dk+j, attended.At(i, j))
			}
		}
	}

	var out mat.Dense
	out.Mul(concat, Wo)
	return &out
}

// ---------- Model ----------

type Config struct {
	DModel    int `json:"d_model"`
	NHeads    int `json:"n_heads"`
	NLayers   int `json:"n_layers"`
	DFF       int `json:"d_ff"`
	MaxSeqLen int `json:"max_seq_len"`
}

// blockJSON mirrors the trainer's export schema.
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
	Config     Config      `json:"config"`
	Vocab      []string    `json:"vocab"`
	TokenEmbed [][]float64 `json:"token_embed"`
	PosEmbed   [][]float64 `json:"pos_embed"`
	Blocks     []blockJSON `json:"blocks"`
	LnFW       []float64   `json:"ln_f_w"`
	LnFB       []float64   `json:"ln_f_b"`
	LmHead     [][]float64 `json:"lm_head"`
}

// Block holds one transformer block's weights as dense matrices.
type Block struct {
	Ln1W, Ln1B     []float64
	Wq, Wk, Wv, Wo *mat.Dense
	Ln2W, Ln2B     []float64
	Fc1W           *mat.Dense
	Fc1B           []float64
	Fc2W           *mat.Dense
	Fc2B           []float64
}

type Model struct {
	Cfg            Config
	Vocab          []string
	Stoi           map[string]int
	TokEmb, PosEmb [][]float64
	Blocks         []*Block
	LnFW, LnFB     []float64
	LMHead         *mat.Dense // [vocab, d_model]
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

func loadModel(path string) (*Model, error) {
	f, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var w weightsJSON
	if err := json.Unmarshal(f, &w); err != nil {
		return nil, err
	}

	blocks := make([]*Block, len(w.Blocks))
	for i, b := range w.Blocks {
		blocks[i] = &Block{
			Ln1W: b.Ln1W, Ln1B: b.Ln1B,
			Wq: toDense(b.Wq), Wk: toDense(b.Wk),
			Wv: toDense(b.Wv), Wo: toDense(b.Wo),
			Ln2W: b.Ln2W, Ln2B: b.Ln2B,
			Fc1W: toDense(b.Fc1W), Fc1B: b.Fc1B,
			Fc2W: toDense(b.Fc2W), Fc2B: b.Fc2B,
		}
	}

	stoi := make(map[string]int, len(w.Vocab))
	for i, v := range w.Vocab {
		stoi[v] = i
	}

	return &Model{
		Cfg:    w.Config,
		Vocab:  w.Vocab,
		Stoi:   stoi,
		TokEmb: w.TokenEmbed,
		PosEmb: w.PosEmbed,
		Blocks: blocks,
		LnFW:   w.LnFW,
		LnFB:   w.LnFB,
		LMHead: toDense(w.LmHead),
	}, nil
}

// forward runs one transformer block.
func (b *Block) forward(x *mat.Dense, nHeads int) *mat.Dense {
	n1 := layerNorm(x, b.Ln1W, b.Ln1B, 1e-5)
	attn := multiHeadAttention(n1, b.Wq, b.Wk, b.Wv, b.Wo, nHeads)

	var xMid mat.Dense
	xMid.Add(x, attn)

	n2 := layerNorm(&xMid, b.Ln2W, b.Ln2B, 1e-5)
	var z1 mat.Dense
	z1.Mul(n2, b.Fc1W)
	addRowVec(&z1, b.Fc1B)
	g := gelu(&z1)
	var ff mat.Dense
	ff.Mul(g, b.Fc2W)
	addRowVec(&ff, b.Fc2B)

	var xOut mat.Dense
	xOut.Add(&xMid, &ff)
	return &xOut
}

// lastLogits runs the full stack and returns the logits at the final position.
func (m *Model) lastLogits(ids []int) []float64 {
	T := len(ids)
	d := m.Cfg.DModel

	x := mat.NewDense(T, d, nil)
	for i, id := range ids {
		for j := 0; j < d; j++ {
			x.Set(i, j, m.TokEmb[id][j]+m.PosEmb[i][j])
		}
	}

	for _, blk := range m.Blocks {
		x = blk.forward(x, m.Cfg.NHeads)
	}

	y := layerNorm(x, m.LnFW, m.LnFB, 1e-5)
	last := mat.Row(nil, T-1, y)

	vocabSize, _ := m.LMHead.Dims()
	logits := make([]float64, vocabSize)
	for v := 0; v < vocabSize; v++ {
		s := 0.0
		for j := 0; j < d; j++ {
			s += last[j] * m.LMHead.At(v, j)
		}
		logits[v] = s
	}
	return logits
}

func (m *Model) encode(s string) []int {
	ids := make([]int, 0, len(s))
	for _, r := range strings.ToLower(s) {
		if id, ok := m.Stoi[string(r)]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

// ---------- Sampling ----------

// sample draws a token id using temperature + top-k. temp <= 0 is greedy.
func sample(logits []float64, temp float64, topK int, rng *rand.Rand) int {
	if temp <= 0 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return best
	}

	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = v / temp
	}

	if topK > 0 && topK < len(scaled) {
		order := make([]int, len(scaled))
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool { return scaled[order[a]] > scaled[order[b]] })
		keep := make(map[int]bool, topK)
		for i := 0; i < topK; i++ {
			keep[order[i]] = true
		}
		for i := range scaled {
			if !keep[i] {
				scaled[i] = math.Inf(-1)
			}
		}
	}

	probs := softmaxRow(scaled)
	r := rng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1
}

// ---------- Generation (the query feature) ----------

func (m *Model) generate(prompt string, maxNew int, temp float64, topK int, rng *rand.Rand) string {
	ids := m.encode(prompt)
	if len(ids) == 0 {
		ids = m.encode(" ")
	}
	maxSeq := m.Cfg.MaxSeqLen
	eos, hasEOS := m.Stoi[eosTok]

	for n := 0; n < maxNew; n++ {
		ctx := ids
		if len(ctx) > maxSeq {
			ctx = ctx[len(ctx)-maxSeq:]
		}
		next := sample(m.lastLogits(ctx), temp, topK, rng)
		if hasEOS && next == eos {
			break
		}
		ids = append(ids, next)
	}

	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(m.Vocab[id])
	}
	return sb.String()
}

func (m *Model) repl(rng *rand.Rand) {
	sc := bufio.NewScanner(os.Stdin)
	fmt.Println("\nInteractive mode — type a prompt (empty line to quit):")
	fmt.Print("> ")
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			break
		}
		fmt.Printf("%q\n", m.generate(line, 300, genTemp, genTopK, rng))
		fmt.Print("> ")
	}
}

func main() {
	m, err := loadModel("tiny_english_gpt.json")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded model: vocab=%d  d_model=%d  layers=%d  d_ff=%d\n",
		len(m.Vocab), m.Cfg.DModel, m.Cfg.NLayers, m.Cfg.DFF)

	rng := rand.New(rand.NewSource(1))
	fmt.Printf("\nGeneration (temp=%g, top-k=%d):\n", genTemp, genTopK)
	for _, p := range []string{"once upon a time", "the little girl", "one day"} {
		fmt.Printf("%-20q → %q\n", p, m.generate(p, 200, genTemp, genTopK, rng))
	}

	if len(os.Args) > 1 && os.Args[1] == "-i" {
		m.repl(rng)
	}
}
