#pragma once

#include <storage_gds.cuh>
#include <utils.cuh>

#include <memory>
#include <vector>

class GDSLayer {
 public:
  GDSLayer() {}
  GDSLayer(const GDSLayer &other) = delete;
  GDSLayer(GDSLayer &&other) = delete;
  GDSLayer &operator=(const GDSLayer &other) = delete;
  GDSLayer &operator=(GDSLayer &&other) = delete;

  // connect to next GDSLayer
  GDSLayer &connect(GDSLayer &next_layer) {
    this->next = &next_layer;
    next_layer.pre = this;

    return next_layer;
  }

  virtual void forward() { throw std::runtime_error("not implement error"); };
  virtual void backward() { throw std::runtime_error("not implement error"); };

  // return pointer of weights and grads
  virtual std::vector<std::pair<GDSStorage *, GDSStorage *>> parameters() {
    throw std::runtime_error("not implement error");
  };

  virtual GDSStorage *get_grad() { return this->grad.get(); }
  virtual GDSStorage *get_output() { return this->output.get(); }

 protected:
  GDSLayer *pre;
  GDSLayer *next;

  // inputs grad and GDSLayer output
  std::unique_ptr<GDSStorage> grad;
  std::unique_ptr<GDSStorage> output;
};