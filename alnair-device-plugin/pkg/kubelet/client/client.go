package client

import (
	"encoding/json"
	"fmt"
	"io"
	v1 "k8s.io/api/core/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	"net/http"
	"time"
)

type KubeletClientConfig struct {
	// Address specifies the kubelet address
	Address string

	// Port specifies the default port - used if no information about Kubelet port can be found in Node.NodeStatus.DaemonEndpoints.
	Port uint

	// TLSClientConfig contains settings to enable transport layer security
	restclient.TLSClientConfig

	// Server requires Bearer authentication
	BearerToken string

	// HTTPTimeout is used by the client to timeout http requests to Kubelet.
	HTTPTimeout time.Duration
}

type KubeletClient struct {
	defaultPort uint
	host        string
	client      *http.Client
}

func NewKubeletClient(config *KubeletClientConfig) (*KubeletClient, error) {
	trans, err := makeTransport(config, true)
	if err != nil {
		return nil, err
	}
	client := &http.Client{
		Transport: trans,
		Timeout:   config.HTTPTimeout,
	}
	return &KubeletClient{
		host:        config.Address,
		defaultPort: config.Port,
		client:      client,
	}, nil
}

func ReadAll(r io.Reader) ([]byte, error) {
	b := make([]byte, 0, 512)
	for {
		if len(b) == cap(b) {
			// Add more capacity (let append pick how much).
			b = append(b, 0)[:len(b)]
		}
		n, err := r.Read(b[len(b):cap(b)])
		b = b[:len(b)+n]
		if err != nil {
			if err == io.EOF {
				err = nil
			}
			return b, err
		}
	}
}

func (k *KubeletClient) GetNodeRunningPods() (*v1.PodList, error) {
	resp, err := k.client.Get(fmt.Sprintf("https://%v:%d/pods/", k.host, k.defaultPort))
	if err != nil {
		return nil, err
	}

	body, err := ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	podLists := &v1.PodList{}
	if err = json.Unmarshal(body, &podLists); err != nil {
		return nil, err
	}
	return podLists, err
}