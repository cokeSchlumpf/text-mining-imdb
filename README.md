# 30300 Text Mining - IMDB Moview Reviews Sentiment Analysis

## Getting Started

Download the [dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and place it in `data` directory. 

## Summary of metrics

```
working tree:
	model/metrics.json:
		{
		  "lr_metrics": {
		    "confusion_matrix": [
		      [
		        2408,
		        735
		      ],
		      [
		        584,
		        2523
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.8048128342245989,
		        "recall": 0.7661469933184856,
		        "f1-score": 0.7850040749796251,
		        "support": 3143
		      },
		      "1": {
		        "precision": 0.7744014732965009,
		        "recall": 0.8120373350498874,
		        "f1-score": 0.7927729772191673,
		        "support": 3107
		      },
		      "accuracy": 0.78896,
		      "macro avg": {
		        "precision": 0.7896071537605499,
		        "recall": 0.7890921641841865,
		        "f1-score": 0.7888885260993962,
		        "support": 6250
		      },
		      "weighted avg": {
		        "precision": 0.7896947384800229,
		        "recall": 0.78896,
		        "f1-score": 0.7888661516609463,
		        "support": 6250
		      }
		    },
		    "accuracy_score": 0.78896
		  }
		}
master:
	model/metrics.json:
		{
		  "lr_metrics": {
		    "confusion_matrix": [
		      [
		        2408,
		        735
		      ],
		      [
		        584,
		        2523
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.8048128342245989,
		        "recall": 0.7661469933184856,
		        "f1-score": 0.7850040749796251,
		        "support": 3143
		      },
		      "1": {
		        "precision": 0.7744014732965009,
		        "recall": 0.8120373350498874,
		        "f1-score": 0.7927729772191673,
		        "support": 3107
		      },
		      "accuracy": 0.78896,
		      "macro avg": {
		        "precision": 0.7896071537605499,
		        "recall": 0.7890921641841865,
		        "f1-score": 0.7888885260993962,
		        "support": 6250
		      },
		      "weighted avg": {
		        "precision": 0.7896947384800229,
		        "recall": 0.78896,
		        "f1-score": 0.7888661516609463,
		        "support": 6250
		      }
		    },
		    "accuracy_score": 0.78896
		  }
		}
exp/baseline:
	model/metrics.json:
		{
		  "confusion_matrix": [
		    [
		      7883,
		      4617
		    ],
		    [
		      8007,
		      4493
		    ]
		  ],
		  "classification_report": {
		    "0": {
		      "precision": 0.4960981749528005,
		      "recall": 0.63064,
		      "f1-score": 0.5553363860514265,
		      "support": 12500
		    },
		    "1": {
		      "precision": 0.4931942919868277,
		      "recall": 0.35944,
		      "f1-score": 0.41582600647848217,
		      "support": 12500
		    },
		    "accuracy": 0.49504,
		    "macro avg": {
		      "precision": 0.49464623346981407,
		      "recall": 0.49504,
		      "f1-score": 0.48558119626495433,
		      "support": 25000
		    },
		    "weighted avg": {
		      "precision": 0.494646233469814,
		      "recall": 0.49504,
		      "f1-score": 0.48558119626495433,
		      "support": 25000
		    }
		  },
		  "accuracy_score": 0.49504
		}
exp/logistic-regression:
	model/metrics.json:
		{
		  "lr_metrics": {
		    "confusion_matrix": [
		      [
		        8415,
		        4085
		      ],
		      [
		        4190,
		        8310
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.6675922253074177,
		        "recall": 0.6732,
		        "f1-score": 0.6703843855805617,
		        "support": 12500
		      },
		      "1": {
		        "precision": 0.6704316256555063,
		        "recall": 0.6648,
		        "f1-score": 0.6676039365334404,
		        "support": 12500
		      },
		      "accuracy": 0.669,
		      "macro avg": {
		        "precision": 0.6690119254814619,
		        "recall": 0.669,
		        "f1-score": 0.668994161057001,
		        "support": 25000
		      },
		      "weighted avg": {
		        "precision": 0.669011925481462,
		        "recall": 0.669,
		        "f1-score": 0.668994161057001,
		        "support": 25000
		      }
		    },
		    "accuracy_score": 0.669
		  }
		}
logistic-regression:
	model/metrics.json:
		{
		  "lr_metrics": {
		    "confusion_matrix": [
		      [
		        8415,
		        4085
		      ],
		      [
		        4190,
		        8310
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.6675922253074177,
		        "recall": 0.6732,
		        "f1-score": 0.6703843855805617,
		        "support": 12500
		      },
		      "1": {
		        "precision": 0.6704316256555063,
		        "recall": 0.6648,
		        "f1-score": 0.6676039365334404,
		        "support": 12500
		      },
		      "accuracy": 0.669,
		      "macro avg": {
		        "precision": 0.6690119254814619,
		        "recall": 0.669,
		        "f1-score": 0.668994161057001,
		        "support": 25000
		      },
		      "weighted avg": {
		        "precision": 0.669011925481462,
		        "recall": 0.669,
		        "f1-score": 0.668994161057001,
		        "support": 25000
		      }
		    },
		    "accuracy_score": 0.669
		  }
		}
logistic-regression-hot-encoding:
	model/metrics.json:
		{
		  "lr_metrics": {
		    "confusion_matrix": [
		      [
		        2408,
		        735
		      ],
		      [
		        584,
		        2523
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.8048128342245989,
		        "recall": 0.7661469933184856,
		        "f1-score": 0.7850040749796251,
		        "support": 3143
		      },
		      "1": {
		        "precision": 0.7744014732965009,
		        "recall": 0.8120373350498874,
		        "f1-score": 0.7927729772191673,
		        "support": 3107
		      },
		      "accuracy": 0.78896,
		      "macro avg": {
		        "precision": 0.7896071537605499,
		        "recall": 0.7890921641841865,
		        "f1-score": 0.7888885260993962,
		        "support": 6250
		      },
		      "weighted avg": {
		        "precision": 0.7896947384800229,
		        "recall": 0.78896,
		        "f1-score": 0.7888661516609463,
		        "support": 6250
		      }
		    },
		    "accuracy_score": 0.78896
		  }
		}
random-forest-stemming:
	model/metrics.json:
		{
		  "rf_metrics": {
		    "confusion_matrix": [
		      [
		        8322,
		        4178
		      ],
		      [
		        5238,
		        7262
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.613716814159292,
		        "recall": 0.66576,
		        "f1-score": 0.6386799693016116,
		        "support": 12500
		      },
		      "1": {
		        "precision": 0.6347902097902098,
		        "recall": 0.58096,
		        "f1-score": 0.6066833751044277,
		        "support": 12500
		      },
		      "accuracy": 0.62336,
		      "macro avg": {
		        "precision": 0.624253511974751,
		        "recall": 0.62336,
		        "f1-score": 0.6226816722030197,
		        "support": 25000
		      },
		      "weighted avg": {
		        "precision": 0.624253511974751,
		        "recall": 0.62336,
		        "f1-score": 0.6226816722030197,
		        "support": 25000
		      }
		    },
		    "accuracy_score": 0.62336
		  }
		}
svm:
	model/metrics.json:
		{
		  "svc_metrics": {
		    "confusion_matrix": [
		      [
		        4539,
		        7961
		      ],
		      [
		        1203,
		        11297
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.790491118077325,
		        "recall": 0.36312,
		        "f1-score": 0.4976428023243065,
		        "support": 12500
		      },
		      "1": {
		        "precision": 0.5866133554886281,
		        "recall": 0.90376,
		        "f1-score": 0.7114427860696517,
		        "support": 12500
		      },
		      "accuracy": 0.63344,
		      "macro avg": {
		        "precision": 0.6885522367829766,
		        "recall": 0.63344,
		        "f1-score": 0.6045427941969791,
		        "support": 25000
		      },
		      "weighted avg": {
		        "precision": 0.6885522367829765,
		        "recall": 0.63344,
		        "f1-score": 0.6045427941969792,
		        "support": 25000
		      }
		    },
		    "accuracy_score": 0.63344
		  }
		}
unigrams-and-bigrams:
	model/metrics.json:
		{
		  "lr_metrics": {
		    "confusion_matrix": [
		      [
		        7466,
		        5034
		      ],
		      [
		        6388,
		        6112
		      ]
		    ],
		    "classification_report": {
		      "0": {
		        "precision": 0.5389057311967663,
		        "recall": 0.59728,
		        "f1-score": 0.5665933065189345,
		        "support": 12500
		      },
		      "1": {
		        "precision": 0.5483581553920689,
		        "recall": 0.48896,
		        "f1-score": 0.5169584707772985,
		        "support": 12500
		      },
		      "accuracy": 0.54312,
		      "macro avg": {
		        "precision": 0.5436319432944177,
		        "recall": 0.54312,
		        "f1-score": 0.5417758886481165,
		        "support": 25000
		      },
		      "weighted avg": {
		        "precision": 0.5436319432944176,
		        "recall": 0.54312,
		        "f1-score": 0.5417758886481165,
		        "support": 25000
		      }
		    },
		    "accuracy_score": 0.54312
		  }
		}
```