#pragma once

#include <bf/fac_streamer.h>
#include <bf/interval.h>
#include <bf/points.h>

BfPoints1 *bfLboEigsToFreqs(BfVecReal const *Lam);

typedef struct {
  bool success;
  BfReal eigenbandTime;
  BfReal totalTime;
  BfInterval freqBand;
} BfLboFeedResult;
BfLboFeedResult bfLboFeedFacStreamerNextEigenband(BfFacStreamer *facStreamer, BfPoints1 *freqs, BfMat const *L, BfMat const *M);
