#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../include/shimaenaga/booster.h"
#include "../include/shimaenaga/dataset.h"
#include "../include/shimaenaga/config.h"

namespace py = pybind11;
using namespace shimaenaga;

// Helper: ensure C-contiguous float32 array
static py::array_t<float> ensure_float32(py::array X) {
  return py::array_t<float, py::array::c_style | py::array::forcecast>(X);
}

// Helper: ensure C-contiguous float64 array
static py::array_t<double> ensure_float64(py::array X) {
  return py::array_t<double, py::array::c_style | py::array::forcecast>(X);
}

PYBIND11_MODULE(_shimaenaga, m) {
  m.doc() = "Shimaenaga: Attentive GBDT Python bindings";

  // Config
  py::class_<Config>(m, "Config")
    .def(py::init<>())
    .def_readwrite("objective",          &Config::objective)
    .def_readwrite("num_class",          &Config::num_class)
    .def_readwrite("num_iterations",     &Config::num_iterations)
    .def_readwrite("learning_rate",      &Config::learning_rate)
    .def_readwrite("tier",               &Config::tier)
    .def_readwrite("num_tokens",         &Config::num_tokens)
    .def_readwrite("num_heads",          &Config::num_heads)
    .def_readwrite("attention_mode",     &Config::attention_mode)
    .def_readwrite("d_attn",             &Config::d_attn)
    .def_readwrite("eta_attn",           &Config::eta_attn)
    .def_readwrite("attn_mask",          &Config::attn_mask)
    .def_readwrite("token_num_leaves",   &Config::token_num_leaves)
    .def_readwrite("gate_num_leaves",    &Config::gate_num_leaves)
    .def_readwrite("inner_refit_steps",  &Config::inner_refit_steps)
    .def_readwrite("min_data_in_leaf",   &Config::min_data_in_leaf)
    .def_readwrite("lambda_v",           &Config::lambda_v)
    .def_readwrite("lambda_q",           &Config::lambda_q)
    .def_readwrite("lambda_k",           &Config::lambda_k)
    .def_readwrite("lambda_z",           &Config::lambda_z)
    .def_readwrite("lambda_ent",         &Config::lambda_ent)
    .def_readwrite("lambda_div",         &Config::lambda_div)
    .def_readwrite("max_bin",            &Config::max_bin)
    .def_readwrite("early_stopping_rounds", &Config::early_stopping_rounds)
    .def_readwrite("num_threads",        &Config::num_threads)
    .def_readwrite("seed",               &Config::seed)
    .def_static("from_dict", [](const std::map<std::string,std::string>& d) {
      return Config::FromMap(d);
    });

  // Dataset
  py::class_<Dataset, std::shared_ptr<Dataset>>(m, "Dataset")
    .def_static("build",
      [](py::array X_in, py::array y_in,
         py::object weights_in, py::object groups_in,
         const std::map<std::string,std::string>& params) {
        auto X = ensure_float32(X_in);
        auto y = ensure_float32(y_in);
        auto cfg = Config::FromMap(params);

        auto* Xp = X.data();
        auto* yp = y.data();
        data_size_t n = (data_size_t)X.shape(0);
        int nf = (int)X.shape(1);

        const float* wp = nullptr;
        std::vector<float> w_buf;
        if (!weights_in.is_none()) {
          auto wa = ensure_float32(weights_in.cast<py::array>());
          w_buf.assign(wa.data(), wa.data() + wa.size());
          wp = w_buf.data();
        }

        const int32_t* gp = nullptr;
        int ng = 0;
        std::vector<int32_t> g_buf;
        if (!groups_in.is_none()) {
          auto ga = py::array_t<int32_t, py::array::c_style|py::array::forcecast>(
              groups_in.cast<py::array>());
          g_buf.assign(ga.data(), ga.data() + ga.size());
          gp = g_buf.data();
          ng = (int)g_buf.size();
        }

        py::gil_scoped_release release;
        // Return shared_ptr explicitly (not unique_ptr): the class holder
        // below is std::shared_ptr<Dataset>, and letting pybind11 convert
        // a returned unique_ptr into that holder triggers a stack-buffer
        // overflow in its own init_instance<> machinery on this pybind11
        // version.
        return std::shared_ptr<Dataset>(Dataset::Build(Xp, n, nf, yp, wp, gp, ng, cfg));
      },
      py::arg("X"), py::arg("y"),
      py::arg("weights") = py::none(),
      py::arg("groups") = py::none(),
      py::arg("params") = std::map<std::string,std::string>{})
    .def("build_like",
      [](std::shared_ptr<Dataset> train,
         py::array X_in, py::array y_in,
         py::object weights_in, py::object groups_in) {
        auto X = ensure_float32(X_in);
        auto y = ensure_float32(y_in);
        data_size_t n = (data_size_t)X.shape(0);
        const float* wp = nullptr;
        std::vector<float> w_buf;
        if (!weights_in.is_none()) {
          auto wa = ensure_float32(weights_in.cast<py::array>());
          w_buf.assign(wa.data(), wa.data() + wa.size());
          wp = w_buf.data();
        }
        const int32_t* gp = nullptr;
        int ng = 0;
        std::vector<int32_t> g_buf;
        if (!groups_in.is_none()) {
          auto ga = py::array_t<int32_t, py::array::c_style|py::array::forcecast>(
              groups_in.cast<py::array>());
          g_buf.assign(ga.data(), ga.data() + ga.size());
          gp = g_buf.data(); ng = (int)g_buf.size();
        }
        if ((int)X.shape(1) != train->NumFeatures())
          throw std::invalid_argument(
              "build_like: X has a different number of features than the training dataset");
        py::gil_scoped_release release;
        return std::shared_ptr<Dataset>(Dataset::BuildLike(*train, X.data(), n, y.data(), wp, gp, ng));
      },
      py::arg("X"), py::arg("y"),
      py::arg("weights") = py::none(), py::arg("groups") = py::none())
    .def("num_data",     &Dataset::NumData)
    .def("num_features", &Dataset::NumFeatures);

  // Booster
  py::class_<Booster>(m, "Booster")
    .def(py::init([](std::shared_ptr<Dataset> train,
                     const std::map<std::string,std::string>& params) {
      auto cfg = Config::FromMap(params);
      return std::make_unique<Booster>(cfg, std::move(train));
    }), py::arg("train"), py::arg("params") = std::map<std::string,std::string>{})
    .def("add_valid", [](Booster& b, std::shared_ptr<Dataset> valid) {
      b.AddValidData(std::move(valid));
    })
    .def("train", [](Booster& b) {
      py::gil_scoped_release release;
      b.Train();
    })
    .def("predict", [](Booster& b, py::array X_in) {
      auto X = ensure_float32(X_in);
      data_size_t n = (data_size_t)X.shape(0);
      int nf = (int)X.shape(1);
      std::vector<score_t> pred;
      {
        py::gil_scoped_release release;
        pred = b.Predict(X.data(), n, nf);
      }
      int C = b.GetModel().C;
      if (C == 1) {
        py::array_t<double> out({(py::ssize_t)n});
        std::copy(pred.begin(), pred.end(), out.mutable_data());
        return out.cast<py::object>();
      } else {
        py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)C});
        std::copy(pred.begin(), pred.end(), out.mutable_data());
        return out.cast<py::object>();
      }
    })
    .def("predict_proba", [](Booster& b, py::array X_in) {
      auto X = ensure_float32(X_in);
      data_size_t n = (data_size_t)X.shape(0);
      int nf = (int)X.shape(1);
      std::vector<score_t> pred;
      {
        py::gil_scoped_release release;
        pred = b.Predict(X.data(), n, nf);
      }
      int C = b.GetModel().C;
      if (C == 1) {
        // Binary: sigmoid
        py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)2});
        double* p = out.mutable_data();
        for (data_size_t i = 0; i < n; ++i) {
          double pos = 1.0 / (1.0 + std::exp(-pred[i]));
          p[i*2]   = 1.0 - pos;
          p[i*2+1] = pos;
        }
        return out.cast<py::object>();
      } else {
        // Multiclass: softmax
        py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)C});
        double* p = out.mutable_data();
        for (data_size_t i = 0; i < n; ++i) {
          double mx = *std::max_element(pred.begin() + i*C, pred.begin() + (i+1)*C);
          double sum = 0.0;
          for (int k = 0; k < C; ++k) { p[i*C+k] = std::exp(pred[i*C+k]-mx); sum += p[i*C+k]; }
          for (int k = 0; k < C; ++k) p[i*C+k] /= sum;
        }
        return out.cast<py::object>();
      }
    })
    .def("predict_contrib", [](Booster& b, py::array X_in) {
      auto X = ensure_float32(X_in);
      data_size_t n = (data_size_t)X.shape(0);
      int nf = (int)X.shape(1);
      std::vector<score_t> pred;
      std::vector<float> beta;
      {
        py::gil_scoped_release release;
        pred = b.PredictContrib(X.data(), n, nf, &beta);
      }
      int C = b.GetModel().C;
      const auto& blocks = b.GetModel().blocks;
      int P = blocks.empty() ? 1 : blocks[0].P;
      py::object scores;
      if (C == 1) {
        py::array_t<double> out({(py::ssize_t)n});
        std::copy(pred.begin(), pred.end(), out.mutable_data());
        scores = out.cast<py::object>();
      } else {
        py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)C});
        std::copy(pred.begin(), pred.end(), out.mutable_data());
        scores = out.cast<py::object>();
      }
      py::array_t<float> beta_arr({(py::ssize_t)n, (py::ssize_t)P});
      std::copy(beta.begin(), beta.end(), beta_arr.mutable_data());
      return py::make_tuple(scores, beta_arr);
    })
    .def("num_tokens", [](const Booster& b) {
      const auto& blocks = b.GetModel().blocks;
      return blocks.empty() ? 0 : blocks[0].P;
    })
    .def("save_model", &Booster::SaveModel)
    .def("load_model", &Booster::LoadModel)
    .def("best_iteration", &Booster::BestIteration)
    .def("num_iterations", [](const Booster& b) {
      return (int)b.GetModel().blocks.size();
    });
}
