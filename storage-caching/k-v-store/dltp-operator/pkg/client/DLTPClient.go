package client

import (
	"context"
	"errors"
	"log"
	"os"
	"time"

	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator/api/v1alpha1"
	pb "github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator/pkg/grpc"
	"google.golang.org/grpc"
)

type DLTSecret struct {
	username string
	password string
}

type DLTPClient struct {
	server string
	port   int
	pod    v1alpha1.DLTPod

	conn_client      pb.ConnectionClient
	hb_client        pb.HeartbeatClient
	log_client       pb.LoggerClient
	reg_client       pb.RegistrationClient
	cachemiss_client pb.CacheMissClient
}

// Loggers
var (
	Info    *log.Logger
	Warning *log.Logger
	Error   *log.Logger
)

// DLTPod defines DLTPod API.
// type DLTPClient interface {
// 	GenerateToken(userName, tokenName string) (*UserToken, error)
// 	Info() (*dltp.ExecutorResponse, error)
// 	SafeRestart() error
// 	Create(config string, options ...interface{}) (*dltp.Job, error)
// 	CreateOrUpdate(config, jobName string) (*dltp.Job, bool, error)
// 	Delete(name string) (bool, error)

// 	GetJob(id string, parentIDs ...string) (*dltp.Job, error)
// 	UpdateJob(id string, parentIDs ...string) (*dltp.Job, error)
// 	GetAllJobNames() ([]dltp.InnerJob, error)
// 	ExecuteScript(groovyScript string) (logs string, err error)
// }

func (dltc *DLTPClient) initLoggers() {
	Info = log.New(os.Stdout,
		"INFO: ",
		log.Ldate|log.Ltime|log.Lshortfile)

	Warning = log.New(os.Stdout,
		"WARNING: ",
		log.Ldate|log.Ltime|log.Lshortfile)

	Error = log.New(os.Stderr,
		"ERROR: ",
		log.Ldate|log.Ltime|log.Lshortfile)
}

// Init Method.
func (dltc *DLTPClient) Init(ctx context.Context) (*DLTPClient, error) {
	dltc.initLoggers()

	// Check channel
	channel, err := grpc.Dial(dltc.server, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		errors.New("Failed to create gRPC channel.")
		return nil, err
	}
	defer channel.Close()

	// Create Connection
	dltc.conn_client = pb.NewConnectionClient(channel)
	dltc.hb_client = pb.NewHeartbeatClient(channel)
	dltc.log_client = pb.NewLoggerClient(channel)
	dltc.reg_client = pb.NewRegistrationClient(channel)
	dltc.cachemiss_client = pb.NewCacheMissClient(channel)

	conn_ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	conn_req := &pb.ConnectRequest{
		Cred: &pb.Credential{
			Username: nil,
			Password: nil,
		},
		CreateUser: true,
	}
	resp, err := dltc.conn_client.Connect(conn_ctx, conn_req)
	if err != nil {
		errors.New("Failed to connect to gRPC server")
		return nil, err
	}

	// Register Jobs
	for i, job := range dltc.pod.Spec.Jobs {
		reg_job := pb.RegisterRequest{}
		resp, err := dltc.reg_client.Register(reg_job)
	}
	return dltc, nil
}

// CreateOrUpdateJob creates or updates a job from config.
func (dltp *DLTPClient) UpdateJob(config, jobName string) (updated bool, err error) {
	// TODO
}

func isNotFoundError(err error) bool {
	if err != nil {
		return err.Error() == errorNotFound.Error()
	}
	return false
}
