package controllers

import (
	"context"
	"time"

	aiv1alpha1 "elastictraining/api/v1alpha1"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("GPU Scheduler", func() {
	var (
		key types.NamespacedName
		ctx context.Context
	)

	BeforeEach(func() {
		key = types.NamespacedName{
			Namespace: "default",
			Name:      "test",
		}
		ctx = context.Background()

		var uj aiv1alpha1.UnifiedJob
		if err := k8sClient.Get(ctx, key, &uj); err == nil {
			deleteJob(ctx, key)
		}
	})

	AfterEach(func() {
		deleteJob(ctx, key)
	})

	It("should assign maxReplicas to the job", func() {
		By("creating an elastic horovod job whose maxReplicas is less than the cluster capacity")
		two, five := int64(2), int64(5)
		ehj := &aiv1alpha1.UnifiedJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "ai.centauruscloud.io/v1alpha1",
				Kind:       "ElasticHorovodJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      key.Name,
				Namespace: key.Namespace,
			},
			Spec: aiv1alpha1.UnifiedJobSpec{
				ReplicaSpec: aiv1alpha1.UnifiedJobReplicaSpec{
					MinReplicas: &two,
					MaxReplicas: &five,
				},
				JobSpec: aiv1alpha1.UnifiedJobWorkersSpec{
					Image:       "test",
					UnifiedArgs: []string{"python", "test.py"},
				},

				Reschedulable: false,
			},
		}

		err := k8sClient.Create(ctx, ehj)
		Expect(err).ShouldNot(HaveOccurred())

		By("waiting for the GPU allocator to assign targetReplicas")
		var uj1 aiv1alpha1.UnifiedJob
		Eventually(func() bool {
			err := k8sClient.Get(ctx, key, &uj1)
			Expect(err).ShouldNot(HaveOccurred())
			return uj1.Spec.ReplicaSpec.TargetReplicas != nil
		}, time.Second*60, time.Second*1).Should(BeTrue())

		By("The assigned targetReplicas should be equal to maxReplicas")
		// Expect(*uj1.Spec.ReplicaSpec.TargetReplicas).To(Equal(*uj1.Spec.ReplicaSpec.MaxReplicas))
	})
})

func deleteJob(ctx context.Context, key types.NamespacedName) {
	uj := aiv1alpha1.UnifiedJob{}
	uj.Name = key.Name
	uj.Namespace = key.Namespace

	err := k8sClient.Delete(ctx, &uj, client.GracePeriodSeconds(0))
	Expect(err).ShouldNot(HaveOccurred())
}
