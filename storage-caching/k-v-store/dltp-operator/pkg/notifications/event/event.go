package event

import (
	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator/api/v1alpha1"
	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator/pkg/notifications/reason"
)

// Phase defines the context where notification has been generated: base or user.
type Phase string

// StatusColor is useful for better UX.
type StatusColor string

// LoggingLevel is type for selecting different logging levels.
type LoggingLevel string

// Event contains event details which will be sent as a notification.
type Event struct {
	DLTPod v1alpha1.DLTPod
	Phase  Phase
	Level  v1alpha1.NotificationLevel
	Reason reason.Reason
}

const (
	// PhaseBase is core configuration of Jenkins provided by the Operator
	PhaseBase Phase = "base"

	// PhaseUser is user-defined configuration of Jenkins
	PhaseUser Phase = "user"
)
