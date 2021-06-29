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

		var ehj aiv1alpha1.ElasticHorovodJob
		if err := k8sClient.Get(ctx, key, &ehj); err == nil {
			deleteJob(ctx, key)
		}
	})

	AfterEach(func() {
		deleteJob(ctx, key)
	})

	It("should assign maxReplicas to the job", func() {
		By("creating an elastic horovod job whose maxReplicas is less than the cluster capacity")
		two, five := int32(2), int32(5)
		ehj := &aiv1alpha1.ElasticHorovodJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "ai.centauruscloud.io/v1alpha1",
				Kind:       "ElasticHorovodJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      key.Name,
				Namespace: key.Namespace,
			},
			Spec: aiv1alpha1.ElasticHorovodJobSpec{
				LauncherSpec: aiv1alpha1.HorovodLauncherSpec{
					Image:         "test",
					PythonCommand: []string{"python", "/test.py"},
				},
				WorkersSpec: aiv1alpha1.ElasticHorovodWorkersSpec{
					Image:       "test",
					MinReplicas: &two,
					MaxReplicas: &five,
				},
			},
		}

		err := k8sClient.Create(ctx, ehj)
		Expect(err).ShouldNot(HaveOccurred())

		By("waiting for the GPU allocator to assign targetReplicas")
		var ehj1 aiv1alpha1.ElasticHorovodJob
		Eventually(func() bool {
			err := k8sClient.Get(ctx, key, &ehj1)
			Expect(err).ShouldNot(HaveOccurred())
			return ehj1.Spec.WorkersSpec.TargetReplicas != nil
		}, time.Second*60, time.Second*1).Should(BeTrue())

		By("The assigned targetReplicas should be equal to maxReplicas")
		Expect(*ehj1.Spec.WorkersSpec.TargetReplicas).To(Equal(*ehj1.Spec.WorkersSpec.MaxReplicas))
	})
})

func deleteJob(ctx context.Context, key types.NamespacedName) {
	ehj := aiv1alpha1.ElasticHorovodJob{}
	ehj.Name = key.Name
	ehj.Namespace = key.Namespace

	err := k8sClient.Delete(ctx, &ehj, client.GracePeriodSeconds(0))
	Expect(err).ShouldNot(HaveOccurred())
}
