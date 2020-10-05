package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
)

type organism struct {
	a, b, c, d, e, f float64 // genes
	badness          float64 // opposite of fitness, lower is better
}

// performs a deep copy on an organism and returns a new organism with the same traits
func (o *organism) Copy() *organism {
	return &organism{
		a:       o.a,
		b:       o.b,
		c:       o.c,
		d:       o.d,
		e:       o.e,
		f:       o.f,
		badness: 0,
	}
}

// reproduce takes the average of each weight and assign it to the new organism
func reproduce(organisms []*organism) *organism {
	o := &organism{badness: 0}

	// take the average of each weight and assign it to the new organism
	a, b, c, d, e, f := 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	for _, org := range organisms {
		a += org.a
		b += org.b
		c += org.c
		d += org.d
		e += org.e
		f += org.f
	}
	o.a = a / float64(len(organisms))
	o.b = b / float64(len(organisms))
	o.c = c / float64(len(organisms))
	o.d = d / float64(len(organisms))
	o.e = e / float64(len(organisms))
	o.f = f / float64(len(organisms))

	return o
}

// Mutate changes the weights of the organism using a random number between (-1 and 1) multiplied by the learningRate
// lr is variable so we can easily change the rate of learning
func (o *organism) Mutate(lr float64) {
	// TODO: Change mutation to a more 'elegant' algorithm if necessary
	o.a += (rand.Float64()*2 - 1) * lr
	o.b += (rand.Float64()*2 - 1) * lr
	o.c += (rand.Float64()*2 - 1) * lr
	o.d += (rand.Float64()*2 - 1) * lr
	o.e += (rand.Float64()*2 - 1) * lr
	o.f += (rand.Float64()*2 - 1) * lr
}

// newRandomOrganism creates a new organism with random weights between -10 and 10
func newRandomOrganism() *organism {
	return &organism{
		a:       rand.Float64()*20 - 10,
		b:       rand.Float64()*20 - 10,
		c:       rand.Float64()*20 - 10,
		d:       rand.Float64()*20 - 10,
		e:       rand.Float64()*20 - 10,
		f:       rand.Float64()*20 - 10,
		badness: 0,
	}
}

// Fit takes in xs (inputs) and ys (targets), then it calculates the predictions according to the function:
// f(x)=y. which is: ax^5 + bx^4 + cx^3 + dx^2 + ex + f in this case
// then calculates the squared error, which is assigned to the badness (lower is better)
func (o *organism) Fit(xs, ys []float64, f func(x float64) float64) {
	predictions := make([]float64, len(xs))
	for i := 0; i < len(xs); i++ {
		predictions[i] = f(xs[i])
	}

	se := 0.0 // squared error
	for i := 0; i < len(predictions); i++ {
		se += (predictions[i] - ys[i]) * (predictions[i] - ys[i])
	}
	o.badness = se
}

// bestOrganisms returns the best 'limit' amount of organisms from the organisms slice based on badness
func bestOrganisms(organisms []*organism, limit int) []*organism {
	// sort low to high by 'badness'
	sort.SliceStable(organisms, func(i, j int) bool {
		return organisms[i].badness < organisms[j].badness
	})
	// return everything if the limit is higher than the amount of organisms
	if len(organisms) <= limit {
		return organisms
	}
	// return 0 - limit, which contain the best values since the slice is now ordered by badness low - high
	return organisms[:limit]
}

// data is a struct that holds the xs en ys
type data struct {
	xs []float64
	ys []float64
}

func main() {
	///----Reading Data----///
	f, err := os.Open("./assets/data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	// read the csv with the data points
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	d := data{
		xs: make([]float64, len(records)-1),
		ys: make([]float64, len(records)-1),
	}
	// parse the data from csv to our 'data' struct
	// discard the header (first row) because it has no actual values
	for idx, record := range records[1:] {
		// these should not fail because that means the data source is corrupt
		x, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}
		y, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Fatal(err)
		}

		d.xs[idx] = x
		d.ys[idx] = y
	}

	///----Genetic Algorithm----///
	population := 12000  // 12000 organisms in our population
	limit := 10          // the limit of 'parents' that will be copied to the new generation
	mutationRate := 0.3  // 30% of the new generation mutates (this is very high, but works great for this case)
	learningRate := 0.01 // the rate the 'weights' change inside the organism when it's mutating. high overshoots, low takes longer to learn
	lossThreshold := 0.5 // the loss threshold at which we want to end the learning loop, lower == better, but takes longer

	concurrency := runtime.NumCPU()       // number of concurrent jobs
	batchSize := population / concurrency // size of each batch

	organisms := make([]*organism, population)
	for i := 0; i < len(organisms); i++ {
		organisms[i] = newRandomOrganism()
	}
	var wg sync.WaitGroup

	loss := math.MaxFloat64

	for gen := 0; loss >= lossThreshold; gen++ { // max se of 0.2
		if gen%100 == 0 {
			fmt.Printf("generation: %d, loss: %f\n", gen, loss)
		}
		wg.Add(concurrency)
		// maybe overkill to do this concurrently, but it works ;)
		for c := 0; c < concurrency; c++ {
			go func(c int) {
				defer wg.Done()
				// 0 - batchSize for first batch
				for _, o := range organisms[c*batchSize : c*batchSize+batchSize] {
					// function
					f := func(x float64) float64 {
						return o.a*math.Pow(x, 5) + o.b*math.Pow(x, 4) +
							o.c*math.Pow(x, 3) + o.d*math.Pow(x, 2) +
							o.e*x + o.f
					}
					o.Fit(d.xs, d.ys, f)
				}
			}(c)
		}
		wg.Wait()

		best := bestOrganisms(organisms, limit) // returns ordered slice where first element is best
		loss = best[0].badness                  // set the loss to the badness of the best 'brain' in the population
		idx := 0
		for i := 0; i < population; i++ {
			// 1 parent
			organisms[i] = best[idx].Copy() // add parents to the new generation, make a copy since we're working with pointers
			// since we add the parents we should never see the loss go up, only down or plateau
			idx++
			if idx > limit-1 { // wrap the idx around limit
				idx = 0
			}
		}
		// best := bestOrganisms(organisms, 2)
		// 2 parents
		//for i := 0; i < population; i++ {
		//	organisms[i] = reproduce(best) // takes the average of both parents
		//}

		// mutate the organisms by 'mutationRate'% of the population
		for i := 0; i < population; i++ {
			// spontaneous mutation
			if rand.Float64() < mutationRate {
				organisms[i].Mutate(learningRate) // using learning rate we can change the speed of mutation,
				// high values can overshoot the correct value, while very low values increase the time to learn a lot
			}
		}
	}
	// print the values of the best organism when the loss has passed the threshold,
	// these values should be converged to the values inside the excel document
	// the values should be close to: a: -0.5, b: 2.0, c: -5.6, d: 8.0, e: -5.0, f: -5.0
	best := bestOrganisms(organisms, 1)
	fmt.Println(best[0])
}
