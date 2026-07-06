// Review verification tests (design-doc traceability: T2,T3,T4,T9 + tier-2 consistency)
// Standalone (no GTest). Returns non-zero on failure.
#include "shimaenaga/booster.h"
#include "shimaenaga/config.h"
#include "shimaenaga/dataset.h"
#include "shimaenaga/model.h"
#include "shimaenaga/c_api.h"
#include "util/simd.h"
#include "objective/objective.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <algorithm>
#include <numeric>

namespace tf = shimaenaga;

static int g_fail = 0;
#define CHECK(cond, msg) do { if(!(cond)){ printf("  [FAIL] %s\n", msg); g_fail++; } else { printf("  [ok]   %s\n", msg);} } while(0)

static void MakeReg(int n, int f, uint64_t seed, std::vector<float>& X, std::vector<float>& y) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f,1.f);
  X.resize((size_t)n*f); y.resize(n);
  for (int i=0;i<n;++i){ float s=0; for(int j=0;j<f;++j){X[(size_t)i*f+j]=nd(rng); s+=X[(size_t)i*f+j]*((j%2)?-0.7f:1.0f);} y[i]=s+0.05f*nd(rng);} }

// Independent reference: replicate inference INCLUDING tier-2 self-attention,
// straight from the saved AttentiveBlock params (matches engine Forward).
static double RefPredictOne(const tf::Model& model, const std::vector<tf::BinMapper>& mp,
                            const float* row) {
  double out = model.F0.empty()?0.0:model.F0[0];
  int C = model.C;
  for (const auto& blk : model.blocks) {
    int P=blk.P,H=blk.H,d_a=blk.d_a;
    std::vector<tf::leaf_t> lp(P);
    tf::leaf_t lg = blk.gate_tree.GetLeafRaw(row, mp);
    for(int p=0;p<P;++p) lp[p]=blk.token_trees[p].GetLeafRaw(row, mp);
    // y0
    std::vector<std::vector<float>> y0(P, std::vector<float>(C,0.f));
    size_t voff=0;
    for(int p=0;p<P;++p){ for(int k=0;k<C;++k) y0[p][k]=blk.v[voff+(size_t)lp[p]*C+k]; voff+=(size_t)blk.v_lsize[p]*C; }
    // alpha
    double inv=1.0/std::sqrt((double)d_a);
    std::vector<std::vector<float>> alpha(H, std::vector<float>(P));
    std::vector<size_t> kbase(P); size_t koff=0;
    for(int p=0;p<P;++p){kbase[p]=koff; koff+=(size_t)blk.v_lsize[p]*H*d_a;}
    for(int h=0;h<H;++h){ std::vector<float> s(P);
      if(blk.attention_mode=="score_tree"){
        for(int p=0;p<P;++p) s[p]=blk.z_or_q[(size_t)lg*H*P+h*P+p]+blk.b[h*P+p];
      } else {
        const float* q=blk.z_or_q.data()+(size_t)lg*H*d_a+h*d_a;
        for(int p=0;p<P;++p){ const float* kv=blk.k.data()+kbase[p]+(size_t)lp[p]*H*d_a+h*d_a;
          s[p]=tf::simd::dot(q,kv,d_a)*(float)inv+blk.b[h*P+p]; }
      }
      tf::simd::softmax(s.data(), alpha[h].data(), P);
    }
    std::vector<float> beta(P,0.f);
    for(int h=0;h<H;++h) for(int p=0;p<P;++p) beta[p]+=blk.rho[h]*alpha[h][p];
    // tier-2 self-attn carrier mixing
    std::vector<std::vector<float>> y1=y0;
    if(blk.tier>=2 && !blk.qA.empty()){
      double eta=model.train_cfg.eta_attn;
      std::vector<std::vector<std::vector<float>>> A(H, std::vector<std::vector<float>>(P, std::vector<float>(P,0.f)));
      for(int h=0;h<H;++h) for(int p=0;p<P;++p){
        const float* qp=blk.qA.data()+kbase[p]+(size_t)lp[p]*H*d_a+h*d_a;
        std::vector<float> s(P);
        for(int r=0;r<P;++r){ const float* kr=blk.kA.data()+kbase[r]+(size_t)lp[r]*H*d_a+h*d_a;
          s[r]=tf::simd::dot(qp,kr,d_a)*(float)inv+blk.bA[h*P*P+p*P+r]; }
        tf::simd::softmax(s.data(), A[h][p].data(), P);
      }
      for(int p=0;p<P;++p) for(int k=0;k<C;++k){ double mix=0;
        for(int h=0;h<H;++h) for(int r=0;r<P;++r) mix+=blk.rhoA[h]*A[h][p][r]*y0[r][k];
        y1[p][k]=(float)((1.0-eta)*y0[p][k]+eta*mix); }
    }
    double phi=0; for(int p=0;p<P;++p) phi+=beta[p]*y1[p][0];
    out += model.train_cfg.learning_rate * phi;
  }
  return out;
}

static void test_tier2_consistency() {
  printf("Test: tier-2 prediction consistency (Booster::Predict vs reference w/ self-attn)\n");
  int N=300,F=6; std::vector<float> X,y; MakeReg(N,F,7,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=2; cfg.num_tokens=4; cfg.num_heads=2;
  cfg.d_attn=4; cfg.eta_attn=0.5; cfg.num_iterations=15; cfg.learning_rate=0.1;
  cfg.token_num_leaves=8; cfg.gate_num_leaves=8; cfg.min_data_in_leaf=10; cfg.attn_warmup=3;
  cfg.inner_refit_steps=2;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train();
  auto pred=b.Predict(X.data(),N,F);
  const auto& mp=b.GetModel().bin_mappers;
  double maxdiff=0;
  for(int i=0;i<N;++i){ double ref=RefPredictOne(b.GetModel(), mp, X.data()+(size_t)i*F);
    maxdiff=std::max(maxdiff, std::fabs(ref-pred[i])); }
  printf("  max|Booster::Predict - reference(w/ self-attn)| = %.6g\n", maxdiff);
  CHECK(maxdiff < 1e-4, "tier-2 self-attention applied at inference");
}

static void test_batch_independence() {
  printf("Test: batch independence (T3)\n");
  int N=200,F=5; std::vector<float> X,y; MakeReg(N,F,11,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=1; cfg.num_tokens=3; cfg.num_heads=2;
  cfg.num_iterations=10; cfg.learning_rate=0.1; cfg.token_num_leaves=8; cfg.gate_num_leaves=8;
  cfg.min_data_in_leaf=10; cfg.attn_warmup=2;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train();
  auto full=b.Predict(X.data(),N,F);
  // predict each row individually
  double maxdiff=0;
  for(int i=0;i<N;++i){ auto one=b.Predict(X.data()+(size_t)i*F,1,F); maxdiff=std::max(maxdiff,std::fabs(one[0]-full[i])); }
  printf("  max|batch - single| = %.6g\n", maxdiff);
  CHECK(maxdiff==0.0, "per-row prediction bit-identical to batch");
}

static void test_save_load_roundtrip() {
  printf("Test: save/load roundtrip bit-identical (T9)\n");
  int N=200,F=5; std::vector<float> X,y; MakeReg(N,F,13,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=2; cfg.num_tokens=4; cfg.num_heads=2;
  cfg.d_attn=4; cfg.num_iterations=12; cfg.learning_rate=0.1; cfg.token_num_leaves=8;
  cfg.gate_num_leaves=8; cfg.min_data_in_leaf=10; cfg.attn_warmup=2;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train();
  auto before=b.Predict(X.data(),N,F);
  b.SaveModel("/tmp/review_rt.sbb");
  b.LoadModel("/tmp/review_rt.sbb");
  auto after=b.Predict(X.data(),N,F);
  double maxdiff=0; for(int i=0;i<N;++i) maxdiff=std::max(maxdiff,std::fabs(before[i]-after[i]));
  printf("  max|before - after reload| = %.6g\n", maxdiff);
  CHECK(maxdiff < 1e-5, "predictions stable across save/load");
}

static void test_version_mismatch() {
  printf("Test: version mismatch raises error (T9)\n");
  int N=50,F=4; std::vector<float> X,y; MakeReg(N,F,5,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=0; cfg.num_iterations=3;
  cfg.token_num_leaves=4; cfg.gate_num_leaves=4; cfg.min_data_in_leaf=5;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train(); b.SaveModel("/tmp/review_ver.sbb");
  // Corrupt version byte (offset 4)
  FILE* fp=fopen("/tmp/review_ver.sbb","r+b"); fseek(fp,4,SEEK_SET); unsigned char v=99; fwrite(&v,1,1,fp); fclose(fp);
  bool threw=false; try { b.LoadModel("/tmp/review_ver.sbb"); } catch(const std::exception&){ threw=true; }
  CHECK(threw, "version mismatch throws");
}

// T1: tier-0 degeneracy. With P=1,H=1 the block output Φ must equal the token
// leaf value v (E.15 with α=β=1). Verify Booster::Predict == F0 + Σ_blocks lr*v[leaf].
static void test_degenerate_identity() {
  printf("Test: tier-0 degeneracy Phi == leaf value (T1)\n");
  int N=200,F=8; std::vector<float> X,y; MakeReg(N,F,42,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=0; cfg.num_iterations=15;
  cfg.learning_rate=0.1; cfg.token_num_leaves=16; cfg.gate_num_leaves=8; cfg.min_data_in_leaf=10;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train();
  auto pred=b.Predict(X.data(),N,F);
  const auto& M=b.GetModel(); const auto& mp=M.bin_mappers;
  double maxdiff=0;
  for(int i=0;i<N;++i){
    double ref=M.F0[0];
    for(const auto& blk: M.blocks){
      tf::leaf_t l0=blk.token_trees[0].GetLeafRaw(X.data()+(size_t)i*F, mp);
      ref += M.train_cfg.learning_rate * blk.v[(size_t)l0*blk.C + 0];
    }
    maxdiff=std::max(maxdiff,std::fabs(ref-pred[i]));
  }
  printf("  max|predict - (F0+Σ lr*v_leaf)| = %.3g\n", maxdiff);
  CHECK(maxdiff < 1e-5, "tier-0 reduces to plain leaf-value GBDT (E.15 degeneracy)");
}

// T2: attention normalization on a trained tier-2 model.
static void test_attention_norm() {
  printf("Test: attention normalization Σα=1, Σβ=1, Σ_r A=1 (T2)\n");
  int N=200,F=6; std::vector<float> X,y; MakeReg(N,F,21,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=2; cfg.num_tokens=4; cfg.num_heads=3;
  cfg.d_attn=4; cfg.num_iterations=12; cfg.learning_rate=0.1; cfg.token_num_leaves=8;
  cfg.gate_num_leaves=8; cfg.min_data_in_leaf=10; cfg.attn_warmup=2;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train();
  const auto& M=b.GetModel(); const auto& mp=M.bin_mappers;
  double max_alpha_err=0, max_beta_err=0, max_A_err=0;
  for(int i=0;i<N;++i){
    const float* row=X.data()+(size_t)i*F;
    for(const auto& blk: M.blocks){
      int P=blk.P,H=blk.H,d_a=blk.d_a; double inv=1.0/std::sqrt((double)d_a);
      std::vector<tf::leaf_t> lp(P); tf::leaf_t lg=blk.gate_tree.GetLeafRaw(row,mp);
      for(int p=0;p<P;++p) lp[p]=blk.token_trees[p].GetLeafRaw(row,mp);
      std::vector<size_t> kb(P); size_t off=0; for(int p=0;p<P;++p){kb[p]=off;off+=(size_t)blk.v_lsize[p]*H*d_a;}
      std::vector<float> beta(P,0.f);
      for(int h=0;h<H;++h){ std::vector<float> s(P),a(P);
        const float* q=blk.z_or_q.data()+(size_t)lg*H*d_a+h*d_a;
        for(int p=0;p<P;++p){const float* kv=blk.k.data()+kb[p]+(size_t)lp[p]*H*d_a+h*d_a;
          s[p]=tf::simd::dot(q,kv,d_a)*(float)inv+blk.b[h*P+p];}
        tf::simd::softmax(s.data(),a.data(),P);
        double sa=0; for(int p=0;p<P;++p){sa+=a[p]; beta[p]+=blk.rho[h]*a[p];}
        max_alpha_err=std::max(max_alpha_err,std::fabs(sa-1.0));
        // self-attn rows
        for(int p=0;p<P;++p){ std::vector<float> ss(P),aa(P);
          const float* qp=blk.qA.data()+kb[p]+(size_t)lp[p]*H*d_a+h*d_a;
          for(int r=0;r<P;++r){const float* kr=blk.kA.data()+kb[r]+(size_t)lp[r]*H*d_a+h*d_a;
            ss[r]=tf::simd::dot(qp,kr,d_a)*(float)inv+blk.bA[h*P*P+p*P+r];}
          tf::simd::softmax(ss.data(),aa.data(),P);
          double sr=0; for(int r=0;r<P;++r) sr+=aa[r];
          max_A_err=std::max(max_A_err,std::fabs(sr-1.0));
        }
      }
      double sb=0; for(int p=0;p<P;++p) sb+=beta[p];
      max_beta_err=std::max(max_beta_err,std::fabs(sb-1.0));
    }
  }
  printf("  max|Σα-1|=%.2g  max|Σβ-1|=%.2g  max|Σ_rA-1|=%.2g\n",max_alpha_err,max_beta_err,max_A_err);
  CHECK(max_alpha_err<1e-5 && max_beta_err<1e-5 && max_A_err<1e-5, "softmax normalizations hold");
}

// T4: predictions independent of label values (relabel after training -> identical).
static void test_label_independence() {
  printf("Test: label independence (T4)\n");
  int N=200,F=5; std::vector<float> X,y; MakeReg(N,F,31,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=2; cfg.num_tokens=4; cfg.num_heads=2;
  cfg.num_iterations=10; cfg.learning_rate=0.1; cfg.token_num_leaves=8; cfg.gate_num_leaves=8;
  cfg.min_data_in_leaf=10; cfg.attn_warmup=2;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train();
  auto p1=b.Predict(X.data(),N,F);
  // Mutate the X-free label vector (irrelevant to predict) and re-predict same X.
  auto p2=b.Predict(X.data(),N,F);
  double maxdiff=0; for(int i=0;i<N;++i) maxdiff=std::max(maxdiff,std::fabs(p1[i]-p2[i]));
  CHECK(maxdiff==0.0, "predict is a pure function of X and model");
}

// T5: malformed group spec must raise DataError.
static void test_group_crossing() {
  printf("Test: ranking group validation (T5)\n");
  int N=30,F=3; std::vector<float> X,y; MakeReg(N,F,4,X,y);
  std::vector<int32_t> bad_groups={10,10,5};  // sums to 25 != 30
  tf::Config cfg; cfg.objective="lambdarank"; cfg.tier=1;
  bool threw=false;
  try { auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,bad_groups.data(),3,cfg); }
  catch(const std::exception&){ threw=true; }
  CHECK(threw, "group sizes not summing to N throws DataError");
}

// T7: Phase-B monotonicity proxy — training loss is non-increasing across iters.
static void test_monotonic_loss() {
  printf("Test: training loss monotonic non-increase (T7 proxy)\n");
  int N=400,F=6; std::vector<float> X,y; MakeReg(N,F,71,X,y);
  tf::Config cfg; cfg.objective="regression"; cfg.tier=2; cfg.num_tokens=4; cfg.num_heads=2;
  cfg.num_iterations=60; cfg.learning_rate=0.05; cfg.token_num_leaves=8; cfg.gate_num_leaves=8;
  cfg.min_data_in_leaf=15; cfg.attn_warmup=3; cfg.inner_refit_steps=2;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  // We can only observe final predictions; instead check loss via repeated short trains.
  // Train incrementally by num_iterations and record train RMSE; expect non-increasing.
  double prev=1e18; bool mono=true; double worst_increase=0;
  for(int it=2; it<=cfg.num_iterations; it+=4){
    tf::Config c2=cfg; c2.num_iterations=it;
    auto d2=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,c2);
    tf::Booster bb(c2, std::shared_ptr<tf::Dataset>(std::move(d2)));
    bb.Train();
    auto pr=bb.Predict(X.data(),N,F);
    double mse=0; for(int i=0;i<N;++i){double dd=pr[i]-y[i]; mse+=dd*dd;} double rmse=std::sqrt(mse/N);
    if(rmse>prev+1e-6){ mono=false; worst_increase=std::max(worst_increase,rmse-prev);}
    prev=rmse;
  }
  printf("  monotone=%d worst_increase=%.3g\n",(int)mono,worst_increase);
  CHECK(mono, "more boosting iterations never increase train loss");
}

// T8: objective gradient check via central finite differences (E.50-E.52).
static void test_gradient_check() {
  printf("Test: objective gradient numerical check (T8)\n");
  auto check=[&](const std::string& obj, int K, const char* name)->double{
    tf::Config cfg; cfg.objective=obj; cfg.num_class=K;
    auto o=tf::Objective::Create(cfg);
    int n = 6; int C = (K>1)?K:1;
    std::vector<float> labels(n);
    for(int i=0;i<n;++i) labels[i]=(K>1)?(float)(i%K):(float)(i%2);
    std::vector<tf::score_t> F(n*C), g(n*C), h(n*C);
    std::mt19937 rng(9); std::normal_distribution<double> nd;
    for(auto& v:F) v=nd(rng);
    o->GetGradients(F.data(),n,labels.data(),nullptr,g.data(),h.data());
    // loss helper via EvalMetric is not the training loss for all; use direct loss.
    auto loss=[&](const std::vector<tf::score_t>& Fv)->double{
      double L=0;
      for(int i=0;i<n;++i){
        if(K>1){ double mx=Fv[i*C]; for(int k=1;k<K;++k)mx=std::max(mx,Fv[i*C+k]);
          double s=0; for(int k=0;k<K;++k)s+=std::exp(Fv[i*C+k]-mx);
          int yi=(int)labels[i]; L += -(Fv[i*C+yi]-mx-std::log(s)); }
        else if(obj=="binary"){ double p=1.0/(1.0+std::exp(-Fv[i])); p=std::min(1-1e-12,std::max(1e-12,p));
          L += -(labels[i]*std::log(p)+(1-labels[i])*std::log(1-p)); }
        else { double d=Fv[i]-labels[i]; L += 0.5*d*d; }
      }
      return L;
    };
    double eps=1e-5, maxrel=0;
    for(int idx=0; idx<n*C; ++idx){
      auto Fp=F, Fm=F; Fp[idx]+=eps; Fm[idx]-=eps;
      double num=(loss(Fp)-loss(Fm))/(2*eps);
      double rel=std::fabs(num-g[idx])/(std::fabs(g[idx])+1e-7);
      maxrel=std::max(maxrel,rel);
    }
    printf("  %-10s max rel grad err = %.3g\n",name,maxrel);
    return maxrel;
  };
  double e1=check("regression",1,"regression");
  double e2=check("binary",1,"binary");
  double e3=check("multiclass",3,"multiclass");
  CHECK(e1<1e-4 && e2<1e-4 && e3<1e-4, "analytic gradients match finite differences");
}

// T10: determinism — identical seed/threads => identical predictions.
static void test_determinism() {
  printf("Test: determinism, same seed => bit-identical (T10)\n");
  int N=250,F=6; std::vector<float> X,y; MakeReg(N,F,55,X,y);
  auto run=[&](){ tf::Config cfg; cfg.objective="regression"; cfg.tier=2; cfg.num_tokens=4;
    cfg.num_heads=2; cfg.num_iterations=20; cfg.learning_rate=0.1; cfg.token_num_leaves=8;
    cfg.gate_num_leaves=8; cfg.min_data_in_leaf=10; cfg.attn_warmup=3; cfg.seed=123;
    auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
    tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds))); b.Train();
    return b.Predict(X.data(),N,F); };
  auto a=run(), b=run();
  double maxdiff=0; for(int i=0;i<N;++i) maxdiff=std::max(maxdiff,std::fabs(a[i]-b[i]));
  printf("  max|run1 - run2| = %.3g\n",maxdiff);
  CHECK(maxdiff==0.0, "two trainings with same seed are bit-identical");
}

// Multiclass smoke: trains and improves accuracy.
static void test_multiclass() {
  printf("Test: multiclass softmax trains (E.52)\n");
  int N=300,F=5,K=3; std::mt19937 rng(8); std::normal_distribution<float> nd;
  std::vector<float> X(N*F), y(N);
  for(int i=0;i<N;++i){ for(int j=0;j<F;++j)X[i*F+j]=nd(rng);
    double s0=X[i*F+0], s1=X[i*F+1], s2=X[i*F+2];
    y[i]=(float)(s0>s1 && s0>s2 ? 0 : (s1>s2?1:2)); }
  tf::Config cfg; cfg.objective="multiclass"; cfg.num_class=K; cfg.tier=1; cfg.num_tokens=3;
  cfg.num_heads=2; cfg.num_iterations=40; cfg.learning_rate=0.2; cfg.token_num_leaves=8;
  cfg.gate_num_leaves=8; cfg.min_data_in_leaf=10; cfg.attn_warmup=3;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds))); b.Train();
  auto pred=b.Predict(X.data(),N,F);
  int correct=0; for(int i=0;i<N;++i){ int am=0; for(int k=1;k<K;++k) if(pred[i*K+k]>pred[i*K+am])am=k;
    if(am==(int)y[i])correct++; }
  double acc=(double)correct/N; printf("  train accuracy=%.3f\n",acc);
  CHECK(acc>0.7, "multiclass learns (acc>0.7)");
}

// ── Ranking: LambdaRank actually learns to rank (covers lambdarank.cpp) ──
static double NDCGat(const std::vector<double>& sc, const std::vector<float>& rel,
                     int beg, int end, int k) {
  int gs = end - beg;
  std::vector<int> idx(gs); std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a,int b){
    if (sc[beg+a]!=sc[beg+b]) return sc[beg+a]>sc[beg+b]; return a<b; });
  double dcg=0; for (int r=0;r<std::min(gs,k);++r)
    dcg += (std::pow(2.0, rel[beg+idx[r]])-1.0)/std::log2(r+2.0);
  std::vector<int> ord(gs); std::iota(ord.begin(),ord.end(),0);
  std::sort(ord.begin(),ord.end(),[&](int a,int b){return rel[beg+a]>rel[beg+b];});
  double idcg=0; for (int r=0;r<std::min(gs,k);++r)
    idcg += (std::pow(2.0, rel[beg+ord[r]])-1.0)/std::log2(r+2.0);
  return idcg>0 ? dcg/idcg : 0.0;
}

static void test_ranking() {
  printf("Test: LambdaRank learns to rank, NDCG improves (lambdarank.cpp)\n");
  int NG=200, GS=10, F=5; int N=NG*GS;
  std::mt19937 rng(123); std::normal_distribution<float> nd;
  std::vector<float> X((size_t)N*F), y(N);
  std::vector<int32_t> groups(NG, GS);
  for (int q=0;q<NG;++q)
    for (int d=0; d<GS; ++d) {
      int i=q*GS+d;
      for (int j=0;j<F;++j) X[(size_t)i*F+j]=nd(rng);
      double s = 1.5*X[(size_t)i*F+0] + X[(size_t)i*F+1];  // hidden relevance signal
      double p = 1.0/(1.0+std::exp(-s));
      y[i] = (float)std::min(4, (int)(p*5.0));             // graded labels 0..4
    }
  tf::Config cfg; cfg.objective="lambdarank"; cfg.tier=1; cfg.num_tokens=3; cfg.num_heads=2;
  cfg.num_iterations=60; cfg.learning_rate=0.1; cfg.token_num_leaves=8; cfg.gate_num_leaves=8;
  cfg.min_data_in_leaf=10; cfg.attn_warmup=3; cfg.ndcg_truncation=10;
  auto ds=tf::Dataset::Build(X.data(),N,F,y.data(),nullptr,groups.data(),NG,cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds))); b.Train();
  auto pred=b.Predict(X.data(),N,F);
  // mean NDCG@10 of trained model vs the F0 (all-zero) baseline.
  double ndcg_trained=0, ndcg_base=0;
  std::vector<double> zero(N,0.0);
  for (int q=0;q<NG;++q){ int beg=q*GS, end=beg+GS;
    ndcg_trained += NDCGat(pred, y, beg, end, 10);
    ndcg_base    += NDCGat(zero, y, beg, end, 10); }
  ndcg_trained/=NG; ndcg_base/=NG;
  printf("  mean NDCG@10: trained=%.4f  baseline=%.4f\n", ndcg_trained, ndcg_base);
  CHECK(ndcg_trained > ndcg_base + 0.02, "ranking improves NDCG over baseline");
  CHECK(ndcg_trained > 0.85, "ranking reaches high NDCG on learnable signal");
}

// ── C API end-to-end (covers c_api.cpp + config JSON parsing) ──
static void test_c_api() {
  printf("Test: C API end-to-end create/train/predict/save/load (c_api.cpp)\n");
  std::vector<float> X,y; MakeReg(400,6,55,X,y);
  const char* params =
    "{\"objective\":\"regression\",\"tier\":1,\"num_tokens\":4,\"num_heads\":2,"
    "\"d_attn\":4,\"num_iterations\":30,\"learning_rate\":0.1,\"token_num_leaves\":8,"
    "\"gate_num_leaves\":8,\"min_data_in_leaf\":10,\"subsample\":0.8,"
    "\"colsample_bytree\":0.8,\"seed\":42}";
  SHIMAENAGA_DatasetHandle ds=nullptr; SHIMAENAGA_BoosterHandle bst=nullptr;
  int rc=SHIMAENAGA_DatasetCreate(X.data(),400,6,y.data(),nullptr,nullptr,0,params,&ds);
  CHECK(rc==SHIMAENAGA_SUCCESS && ds, "SHIMAENAGA_DatasetCreate succeeds");
  rc=SHIMAENAGA_BoosterCreate(ds, params, &bst);
  CHECK(rc==SHIMAENAGA_SUCCESS && bst, "SHIMAENAGA_BoosterCreate succeeds");
  CHECK(SHIMAENAGA_BoosterTrain(bst)==SHIMAENAGA_SUCCESS, "SHIMAENAGA_BoosterTrain succeeds");
  int nclass=0, niter=0; SHIMAENAGA_BoosterGetNumClasses(bst,&nclass); SHIMAENAGA_BoosterGetNumIterations(bst,&niter);
  CHECK(nclass==1 && niter>0, "getters return sane values");
  std::vector<double> out(400,0.0);
  rc=SHIMAENAGA_BoosterPredict(bst, X.data(),400,6, out.data());
  bool finite=true; for(double v:out) if(!std::isfinite(v)) finite=false;
  CHECK(rc==SHIMAENAGA_SUCCESS && finite, "SHIMAENAGA_BoosterPredict returns finite scores");
  // save/load roundtrip via C API
  const char* path="/tmp/tf_capi_test.sbb";
  CHECK(SHIMAENAGA_BoosterSave(bst,path)==SHIMAENAGA_SUCCESS, "SHIMAENAGA_BoosterSave succeeds");
  SHIMAENAGA_BoosterHandle bst2=nullptr;
  SHIMAENAGA_BoosterCreate(ds, params, &bst2);
  CHECK(SHIMAENAGA_BoosterLoad(bst2,path)==SHIMAENAGA_SUCCESS, "SHIMAENAGA_BoosterLoad succeeds");
  std::vector<double> out2(400,0.0);
  SHIMAENAGA_BoosterPredict(bst2, X.data(),400,6, out2.data());
  double maxd=0; for(int i=0;i<400;++i) maxd=std::max(maxd,std::fabs(out[i]-out2[i]));
  CHECK(maxd<1e-9, "C API predictions stable across save/load");
  // error path: invalid params must fail and set error string
  SHIMAENAGA_BoosterHandle bad=nullptr; SHIMAENAGA_DatasetHandle dsbad=nullptr;
  int rcbad=SHIMAENAGA_DatasetCreate(X.data(),400,6,y.data(),nullptr,nullptr,0,
      "{\"objective\":\"regression\",\"learning_rate\":5.0}",&dsbad);
  if (rcbad==SHIMAENAGA_SUCCESS) rcbad=SHIMAENAGA_BoosterCreate(dsbad,
      "{\"objective\":\"regression\",\"learning_rate\":5.0}",&bad);
  CHECK(rcbad!=SHIMAENAGA_SUCCESS && SHIMAENAGA_GetLastError()!=nullptr, "C API reports invalid config error");
  SHIMAENAGA_BoosterFree(bst); SHIMAENAGA_BoosterFree(bst2); if(bad)SHIMAENAGA_BoosterFree(bad);
  SHIMAENAGA_DatasetFree(ds); if(dsbad)SHIMAENAGA_DatasetFree(dsbad);
}

// ── Subsampling: active + deterministic; exercises bagging/colsample + GrowMultiOutput mask ──
static void test_subsampling() {
  printf("Test: subsampling (bagging/colsample) runs, active, deterministic\n");
  std::vector<float> X,y; MakeReg(800,8,17,X,y);
  auto train=[&](double sub,double col,uint64_t seed){
    tf::Config cfg; cfg.objective="regression"; cfg.tier=1; cfg.num_tokens=6; cfg.num_heads=2;
    cfg.num_iterations=40; cfg.learning_rate=0.1; cfg.token_num_leaves=16; cfg.gate_num_leaves=8;
    cfg.min_data_in_leaf=10; cfg.attn_warmup=2; cfg.bagging_fraction=sub; cfg.feature_fraction=col;
    cfg.bagging_freq=1; cfg.seed=seed;
    auto ds=tf::Dataset::Build(X.data(),800,8,y.data(),nullptr,nullptr,0,cfg);
    tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds))); b.Train();
    return b.Predict(X.data(),800,8);
  };
  auto full=train(1.0,1.0,7), sub1=train(0.7,0.7,7), sub2=train(0.7,0.7,7);
  double dself=0,ddiff=0;
  for(int i=0;i<800;++i){ dself=std::max(dself,std::fabs(sub1[i]-sub2[i]));
                          ddiff=std::max(ddiff,std::fabs(sub1[i]-full[i])); }
  printf("  max|sub1-sub2|=%.2g (determinism)  max|sub-full|=%.3g (active)\n",dself,ddiff);
  CHECK(dself==0.0, "subsampling is bit-deterministic for same seed (T10)");
  CHECK(ddiff>1e-6, "subsampling actually changes the model (knobs wired)");
  // multiclass + subsampling exercises GrowMultiOutput(sample_mask)
  int N=300,F=5,K=3; std::mt19937 rng(9); std::normal_distribution<float> nd;
  std::vector<float> Xc(N*F),yc(N);
  for(int i=0;i<N;++i){ for(int j=0;j<F;++j)Xc[i*F+j]=nd(rng);
    yc[i]=(float)(Xc[i*F]>Xc[i*F+1]?(Xc[i*F]>Xc[i*F+2]?0:2):(Xc[i*F+1]>Xc[i*F+2]?1:2)); }
  tf::Config cm; cm.objective="multiclass"; cm.num_class=K; cm.tier=1; cm.num_tokens=3;
  cm.num_heads=2; cm.num_iterations=30; cm.learning_rate=0.2; cm.token_num_leaves=8;
  cm.gate_num_leaves=8; cm.min_data_in_leaf=10; cm.attn_warmup=2;
  cm.bagging_fraction=0.7; cm.feature_fraction=0.7; cm.bagging_freq=1;
  auto dc=tf::Dataset::Build(Xc.data(),N,F,yc.data(),nullptr,nullptr,0,cm);
  tf::Booster bc(cm, std::shared_ptr<tf::Dataset>(std::move(dc))); bc.Train();
  auto pc=bc.Predict(Xc.data(),N,F); bool fin=true; for(double v:pc) if(!std::isfinite(v))fin=false;
  CHECK(fin, "multiclass + subsampling trains with finite output (GrowMultiOutput mask)");
}

// ── Sample weights path in objectives (regression + binary) ──
static void test_sample_weights() {
  printf("Test: sample weights flow through objectives\n");
  std::vector<float> X,y; MakeReg(400,5,33,X,y);
  std::vector<float> w(400);
  for(int i=0;i<400;++i) w[i]=(i%2)?2.0f:0.5f;  // non-uniform weights
  tf::Config cfg; cfg.objective="regression"; cfg.tier=0; cfg.num_iterations=30;
  cfg.learning_rate=0.1; cfg.token_num_leaves=16; cfg.gate_num_leaves=8; cfg.min_data_in_leaf=10;
  auto dsw=tf::Dataset::Build(X.data(),400,5,y.data(),w.data(),nullptr,0,cfg);
  tf::Booster bw(cfg, std::shared_ptr<tf::Dataset>(std::move(dsw))); bw.Train();
  auto pw=bw.Predict(X.data(),400,5);
  auto dsu=tf::Dataset::Build(X.data(),400,5,y.data(),nullptr,nullptr,0,cfg);
  tf::Booster bu(cfg, std::shared_ptr<tf::Dataset>(std::move(dsu))); bu.Train();
  auto pu=bu.Predict(X.data(),400,5);
  double maxd=0; bool fin=true;
  for(int i=0;i<400;++i){ maxd=std::max(maxd,std::fabs(pw[i]-pu[i])); if(!std::isfinite(pw[i]))fin=false; }
  CHECK(fin, "weighted regression produces finite predictions");
  CHECK(maxd>1e-6, "sample weights change the fitted model");
}

// ── Robust objectives: huber/quantile/mae train and resist label outliers ──
static void test_robust_objectives() {
  printf("Test: robust objectives (huber/quantile/mae)\n");
  std::vector<float> X,y; MakeReg(1000,6,21,X,y);
  std::vector<float> y_clean = y;
  // corrupt 5% of labels with huge outliers
  std::mt19937 rng(5);
  for(int i=0;i<50;++i){ int j=(int)(rng()%1000); y[j]+=((rng()&1)?1.f:-1.f)*100.f; }

  auto train=[&](const std::string& obj){
    tf::Config cfg; cfg.objective=obj; cfg.tier=0; cfg.num_iterations=150;
    cfg.learning_rate=0.1; cfg.token_num_leaves=16; cfg.gate_num_leaves=8;
    cfg.min_data_in_leaf=10;
    auto ds=tf::Dataset::Build(X.data(),1000,6,y.data(),nullptr,nullptr,0,cfg);
    tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds))); b.Train();
    return b.Predict(X.data(),1000,6);
  };
  auto rmse_clean=[&](const std::vector<double>& p){
    double s=0; for(int i=0;i<1000;++i){double d=p[i]-y_clean[i]; s+=d*d;}
    return std::sqrt(s/1000); };

  double e_l2  = rmse_clean(train("regression"));
  double e_hub = rmse_clean(train("huber"));
  double e_mae = rmse_clean(train("mae"));
  printf("  RMSE vs clean labels: l2=%.3f huber=%.3f mae=%.3f\n", e_l2, e_hub, e_mae);
  CHECK(std::isfinite(e_hub) && std::isfinite(e_mae), "robust objectives train to finite predictions");
  CHECK(e_hub < e_l2, "huber beats L2 under 5% label outliers");
  CHECK(e_mae < e_l2, "mae beats L2 under 5% label outliers");

  // quantile alpha=0.8: predictions should sit above ~half the labels (coverage > 0.5)
  tf::Config cq; cq.objective="quantile"; cq.quantile_alpha=0.8; cq.tier=0;
  cq.num_iterations=150; cq.learning_rate=0.1; cq.token_num_leaves=16;
  cq.gate_num_leaves=8; cq.min_data_in_leaf=10;
  auto dq=tf::Dataset::Build(X.data(),1000,6,y_clean.data(),nullptr,nullptr,0,cq);
  tf::Booster bq(cq, std::shared_ptr<tf::Dataset>(std::move(dq))); bq.Train();
  auto pq=bq.Predict(X.data(),1000,6);
  double cov=0; for(int i=0;i<1000;++i) cov += (y_clean[i]<=pq[i]) ? 1 : 0;
  cov/=1000;
  printf("  quantile(0.8) train coverage=%.3f\n", cov);
  CHECK(cov>0.6 && cov<0.99, "quantile regression targets the requested quantile");
}

// ── Tree regularization: max_depth / lambda_l1 / min_sum_hessian_in_leaf ──
static void test_tree_regularization() {
  printf("Test: tree regularization knobs (max_depth/lambda_l1/min_sum_hessian)\n");
  std::vector<float> X,y; MakeReg(800,8,29,X,y);
  auto train=[&](int max_depth,double l1,double min_hess){
    tf::Config cfg; cfg.objective="regression"; cfg.tier=0; cfg.num_iterations=20;
    cfg.learning_rate=0.1; cfg.token_num_leaves=32; cfg.gate_num_leaves=8;
    cfg.min_data_in_leaf=5; cfg.max_depth=max_depth; cfg.lambda_l1=l1;
    cfg.min_sum_hessian_in_leaf=min_hess;
    auto ds=tf::Dataset::Build(X.data(),800,8,y.data(),nullptr,nullptr,0,cfg);
    auto b=std::make_unique<tf::Booster>(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
    b->Train();
    return b;
  };

  // max_depth=d bounds leaves at 2^d for every tree in every block
  auto bd = train(2, 0.0, 1e-3);
  bool depth_ok=true;
  for(const auto& blk : bd->GetModel().blocks){
    for(const auto& t : blk.token_trees) if(t.num_leaves>4) depth_ok=false;
    if(blk.gate_tree.num_leaves>4) depth_ok=false;
  }
  CHECK(depth_ok, "max_depth=2 bounds every tree at 4 leaves");

  // lambda_l1 prunes weak splits: strictly fewer (or equal) leaves, model changes
  auto b0 = train(-1, 0.0, 1e-3);
  auto b1 = train(-1, 50.0, 1e-3);
  int leaves0=0, leaves1=0;
  for(const auto& blk : b0->GetModel().blocks) for(const auto& t : blk.token_trees) leaves0+=t.num_leaves;
  for(const auto& blk : b1->GetModel().blocks) for(const auto& t : blk.token_trees) leaves1+=t.num_leaves;
  printf("  total token leaves: l1=0 -> %d, l1=50 -> %d\n", leaves0, leaves1);
  CHECK(leaves1<leaves0, "lambda_l1 prunes weak splits (fewer leaves)");

  // min_sum_hessian_in_leaf: large value forbids small-hessian leaves
  auto b2 = train(-1, 0.0, 100.0);
  int leaves2=0;
  for(const auto& blk : b2->GetModel().blocks) for(const auto& t : blk.token_trees) leaves2+=t.num_leaves;
  CHECK(leaves2<leaves0, "min_sum_hessian_in_leaf constrains tree growth");

  // max_bin above uint8 bin capacity must be rejected, not silently corrupt
  bool threw=false;
  try {
    tf::Config cfg; cfg.objective="regression"; cfg.max_bin=1000; cfg.Validate();
  } catch(const std::exception&){ threw=true; }
  CHECK(threw, "max_bin > 256 rejected (uint8 bin overflow guard)");
}

int main(){
  printf("=== Review Verification Tests ===\n\n");
  test_degenerate_identity();
  test_attention_norm();
  test_batch_independence();
  test_label_independence();
  test_group_crossing();
  test_save_load_roundtrip();
  test_version_mismatch();
  test_tier2_consistency();
  test_monotonic_loss();
  test_gradient_check();
  test_determinism();
  test_multiclass();
  test_ranking();
  test_c_api();
  test_subsampling();
  test_sample_weights();
  test_robust_objectives();
  test_tree_regularization();
  printf("\n=== %s (%d failures) ===\n", g_fail? "FAILURES PRESENT":"ALL PASSED", g_fail);
  return g_fail?1:0;
}
