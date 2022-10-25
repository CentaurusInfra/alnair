package cpucollect

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
)

//var _ prometheus.Collector = &cpuCollector{}

// A collector is a prometheus.Collector for Linux CPU stats.
type cpuCollector struct {
	// Possible metric descriptions.
	TimeUserHertzTotal *prometheus.Desc

	//simplify struct https://github.com/mdlayher/talks/tree/6dcf4eed9e605fa32ced003ddd0dad5fdd3f6df6/conferences/2018/kccnceu/cpustat
	// A parameterized function used to gather metrics.
	//stats func() ([]CPUStat, error)
}

// A CPUStat contains statistics for an individual CPU.
type CPUStat struct {
	// The ID of the CPU.
	ID string

	// The time, in USER_HZ (typically 1/100th of a second),
	// spent in each of user, system, and idle modes.
	User, System, Idle int
}

// NewCollector constructs a prometheus collector, must implement Describe and Collect.
func NewCollector(nodeName string) prometheus.Collector {
	return &cpuCollector{
		TimeUserHertzTotal: prometheus.NewDesc(
			// Name of the metric.
			"alnair_cpustat_time_user_hertz_total",
			// The metric's help text.
			"Time in USER_HZ a given CPU spent in a given mode.",
			// The metric's variable label dimensions.
			[]string{"cpu", "mode"},
			// The metric's constant label dimensions.
			prometheus.Labels{"nodeName": nodeName},
		),
	}
}

// Describe implements prometheus.Collector.
func (c *cpuCollector) Describe(ch chan<- *prometheus.Desc) {
	// Gather metadata about each metric.
	ds := []*prometheus.Desc{
		c.TimeUserHertzTotal,
	}

	for _, d := range ds {
		ch <- d
	}
}

// Collect implements prometheus.Collector.
func (c *cpuCollector) Collect(ch chan<- prometheus.Metric) {
	// Take a stats snapshot.  Must be concurrency safe.
	stats, err := GetCpuStat()
	if err != nil {
		// If an error occurs, send an invalid metric to notify
		// Prometheus of the problem.
		ch <- prometheus.NewInvalidMetric(c.TimeUserHertzTotal, err)
		return
	}

	for _, s := range stats {
		tuples := []struct {
			mode string
			v    int
		}{
			{mode: "user", v: s.User},
			{mode: "system", v: s.System},
			{mode: "idle", v: s.Idle},
		}

		for _, t := range tuples {
			// prometheus.Collector implementations should always use
			// "const metric" constructors.
			ch <- prometheus.MustNewConstMetric(
				c.TimeUserHertzTotal,
				prometheus.CounterValue,
				float64(t.v),
				s.ID, t.mode,
			)
		}
	}
}

func GetCpuStat() ([]CPUStat, error) {
	r, err := os.Open("/proc/stat")
	if err != nil {
		return nil, fmt.Errorf("failed to open /proc/stat: %v", err)
	}
	defer r.Close()

	// Skip the first summarized line.
	s := bufio.NewScanner(r)
	s.Scan()

	var stats []CPUStat
	for s.Scan() {
		// Each CPU stats line should have a "cpu" prefix and exactly
		// 11 fields.
		const nFields = 11
		fields := strings.Fields(string(s.Bytes()))
		if len(fields) != nFields {
			continue
		}
		if !strings.HasPrefix(fields[0], "cpu") {
			continue
		}

		// The values we care about (user, system, idle) lie at indices
		// 1, 3, and 4, respectively.  Parse these into the array.
		var times [3]int
		for i, idx := range []int{1, 3, 4} {
			v, err := strconv.Atoi(fields[idx])
			if err != nil {
				return nil, err
			}

			times[i] = v
		}

		stats = append(stats, CPUStat{
			// First field is the CPU's ID.
			ID:     fields[0],
			User:   times[0],
			System: times[1],
			Idle:   times[2],
		})
	}

	// Be sure to check the error!
	if err := s.Err(); err != nil {
		return nil, err
	}

	return stats, nil
}
