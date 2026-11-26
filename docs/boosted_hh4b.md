# Boosted HH4b

Back to the [main README](../README.md).

The full boosted workflow (PocketCoffea runs, closure plots, category optimisation, datacards) is described in the [boosted README of AnalysisConfigs](https://github.com/matteomalucchi/AnalysisConfigs/blob/main/configs/VBF_HH4b_boosted/README.md). This page covers the two DNNs trained here: the multiclass background morphing and the signal-vs-background classifier.

## Multiclass background morphing

The QCD background is estimated with an ABCD method in the plane of the TXbb of the subleading Higgs candidate and the Higgs mass:

|  | mass sideband | mass window |
| --- | --- | --- |
| **high TXbb** | C: `boosted_signal_sideband_region_C` | D: `boosted_signal_region_D` (signal region) |
| **low TXbb** | A: `boosted_control_sideband_region_A` | B: `boosted_control_region_B` |

The morphing DNN is trained from A (denominator) to a neighbouring region (numerator), and applied in PocketCoffea to the region next to D. By default the morphing goes along the TXbb (numerator C, applied to B); with `bkg_morphing_direction: "mass"` in the PocketCoffea config it goes along the mass (numerator B, applied to C).

Since ttbar is not negligible in these regions, the model is a 4-class classifier, so that the ttbar contribution can be subtracted from the data.

### Config

The four classes must have this `lbl` order (the names are only labels):

| `lbl` | Class | Region (TXbb direction) |
| --- | --- | --- |
| 0 | data, numerator | `boosted_signal_sideband_region_C` |
| 1 | data, denominator | `boosted_control_sideband_region_A` |
| 2 | ttbar MC, numerator | `boosted_signal_sideband_region_C` |
| 3 | ttbar MC, denominator | `boosted_control_sideband_region_A` |

Example, shortened from [`boosted_vbf/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_wps.yml`](../configs/hh4b_bkg_reweighting/boosted_vbf/1e-3_e20drop95_minDelta1em5_boosted_reweighting_btagging_reweight_50GeV_lower_band_wps.yml):

```yaml
input_variables: bkg_morphing_dnn_boosted_bdt_inputs_50GeV_wps
ML_model: DNN_multiclass_model
data_format: parquet
data_dirs: [/path/to/<pocketcoffea_output>/]

classes:
  data_A:     # numerator data (region C)
    sample: [DATA_JetMET0_HH4bBoosted, DATA_JetMET1_HH4bBoosted]
    region: [boosted_signal_sideband_region_C]
    dataset: [DATA_JetMET0_HH4bBoosted_2024_EraC, ...]
    lbl: 0
  data_B:     # denominator data (region A)
    sample: [DATA_JetMET0_HH4bBoosted, DATA_JetMET1_HH4bBoosted]
    region: [boosted_control_sideband_region_A]
    dataset: [DATA_JetMET0_HH4bBoosted_2024_EraC, ...]
    lbl: 1
  ttbar_A:    # numerator ttbar (region C)
    sample: [TTto2L2Nu, TTtoLNu2Q, TTto4Q]
    region: [boosted_signal_sideband_region_C]
    dataset: [TTto2L2Nu_2024, TTtoLNu2Q_2024, TTto4Q_2024]
    lbl: 2
  ttbar_B:    # denominator ttbar (region A)
    sample: [TTto2L2Nu, TTtoLNu2Q, TTto4Q]
    region: [boosted_control_sideband_region_A]
    dataset: [TTto2L2Nu_2024, TTtoLNu2Q_2024, TTto4Q_2024]
    lbl: 3
```

### The ttbar fraction alpha

The classes are balanced during the training, so the ttbar outputs have to be scaled back by the real fraction of ttbar in data, $\alpha = N_{t\bar{t}} / N_\mathrm{data}$. `ml_alpha` computes it from the PocketCoffea output of the config (the first `data_dirs`, regions of the classes with `lbl` 0 and 1):

```bash
ml_alpha -c <config> [--json alpha.json]
```

```text
region                            N_data      N_ttbar     alpha    MCstat
boosted_signal_region_D             2361        661.3    0.2801    0.0040
boosted_signal_sideband_region_C   21744       2343.8    0.1078    0.0008
boosted_control_region_B           46741      19818.3    0.4240    0.0011
boosted_control_sideband_region_A 545355      60568.0    0.1111    0.0002
```

The value to use is the alpha of the numerator region: C for the TXbb direction (here `0.1078`), B for the mass direction.

### Training and merging (recommended)

With `--alpha`, `run_training.sh` computes alpha after the trainings (saved in `<outdir>/alpha.json`) and builds the final ONNX model with the ttbar subtraction:

```bash
# 20 trainings with different seeds
./jobs/run_training.sh -c /full/path/boosted_vbf/<config>.yml -o /full/path/out -n 20 -p 4 --alpha

# or a 20-fold training
./jobs/run_training.sh -c /full/path/boosted_vbf/<config>.yml -o /full/path/out -k 20 -g 8 --cpus 40 --alpha
```

The model to use as `bkg_morphing_dnn` in PocketCoffea is:

- with several seeds: `best_models/average_model_from_onnx.onnx`;
- with k-folds: `kfold_onnx/kfold_morphing_ttbar_subtracted_average.onnx` (or `_median.onnx` with `kfold_onnx: median`, see [K-fold training](kfold_training.md)).

By default there is no ttbar subtraction in the denominator (`alpha_den = 0`); add `--alpha-den-computed` to use the computed alpha of the denominator region.

### Merging by hand

If the models were trained without `--alpha`, merge them inside the `best_models` folder of the output:

```bash
cd <output_folder>/best_models
ml_onnx -i ./ -o ./ -ar -mc --config <config_file> -al <alpha_num> -ald <alpha_den>

# e.g.
ml_onnx -i ./ -o ./ -ar -mc -v bkg_morphing_dnn_boosted_bdt_inputs_50GeV -al 0.1078 -ald 0.0
```

| Option | Meaning |
| --- | --- |
| `-ar` | average the weights of all the models in the folder |
| `-mc` | 4-class model, in the class order above |
| `-v` / `--config` | input-variables module of the training, or the config to read it from |
| `-al` | alpha of the numerator region |
| `-ald` | alpha of the denominator region (`0.0`: no subtraction in the denominator) |

The weight of each model is

$$w = \frac{\mathrm{relu}(r_0 - \alpha\, r_2)}{\mathrm{relu}(r_1 - \alpha_\mathrm{den}\, r_3)},$$

i.e. the ttbar-subtracted data in the numerator region over the ttbar-subtracted data in the denominator region, averaged over the models. The result is `best_models/average_model_from_onnx.onnx`.

## Signal-vs-background classifier

The boosted signal-vs-background DNN is a multiclass model with the classes QCD (data), ttbar, VBF and ggF, e.g. [`boosted/1e-3_e20drop95_minDelta1em5_boosted_signal_background_classifier_50GeV.yml`](../configs/hh4b_sig_bkg_classifier/boosted/1e-3_e20drop95_minDelta1em5_boosted_signal_background_classifier_50GeV.yml). Its `data_dirs` should point to the PocketCoffea output produced with the morphing model above.

```bash
./jobs/run_training.sh -c /full/path/hh4b_sig_bkg_classifier/boosted/<config>.yml -o /full/path/out
```

The ONNX model to use as `sig_bkg_dnn` in PocketCoffea is `state_dict/model_best_epoch_<n>.onnx`. PocketCoffea saves its four class scores and the `sig_bkg_dnn_score_ggFvsBKG` / `sig_bkg_dnn_score_VBFvsBKG` discriminants, which are used to optimise the signal categories.
