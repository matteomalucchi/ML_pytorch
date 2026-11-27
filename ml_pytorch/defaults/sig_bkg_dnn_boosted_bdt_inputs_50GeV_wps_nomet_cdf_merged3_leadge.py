# Input variables of the boosted signal-vs-background classifier of the "nomet + merged3 + lead >= sub cut" chain.
# Same 39 variables, in the same order, as `bkg_morphing_boosted_dnn_input_variables_other_group_txbb_cdf_wp_nomet`
# of AnalysisConfigs (configs/HH4b_common/dnn_input_variables.py), the dictionary used as `bkg_morphing_dnn_input_variables` in
# configs/HH4b_common/config_files/boosted_other_group_50GeV_lower_sideband_txbbwps_nomet_cdf_merged3_leadge_model.py
# (checked by loading both: same keys, order and values), and the same as `bkg_morphing_dnn_boosted_bdt_inputs_50GeV_wps_nomet_cdf_merged3_leadge`.
# The only TXbb inputs are working-point indices (no continuous TXbb, no MET):
#   higgs1_btag_wp      = HiggsLeading btagBBTXbb_5wp (nominal index of the leading Higgs, 1..5 after cut_sub_above_lead)
#   higgs2_btag_cdf_wp  = HiggsSubLeading btagBBTXbb_CDF_rescaled_WPs (quantile-mapped subleading index, merged top bins: 1, 2, 3)
from collections import OrderedDict

dnn_input_variables = OrderedDict(
    {
        "n_vbf_cand_jets": ["events", "nJetGoodVBFGood"],
        "vbf_cand_jets_pt_jet0": ["JetGoodVBFGood:0", "pt"],
        "vbf_cand_jets_eta_jet0": ["JetGoodVBFGood:0", "eta"],
        "vbf_cand_jets_phi_jet0": ["JetGoodVBFGood:0", "phi"],
        "vbf_cand_jets_mass_jet0": ["JetGoodVBFGood:0", "mass"],
        "vbf_cand_jets_pt_jet1": ["JetGoodVBFGood:1", "pt"],
        "vbf_cand_jets_eta_jet1": ["JetGoodVBFGood:1", "eta"],
        "vbf_cand_jets_phi_jet1": ["JetGoodVBFGood:1", "phi"],
        "vbf_cand_jets_mass_jet1": ["JetGoodVBFGood:1", "mass"],
        "HT": ["events", "HT"],
        "hh_vec_pt": ["HH", "pt"],
        "hh_vec_eta": ["HH", "eta"],
        "hh_vec_mass": ["HH", "mass"],
        "hh_vec_phi": ["HH", "phi"],
        "hh_vec_DeltaR": ["HH", "dR"],
        "hh_vec_DeltaPhi": ["HH", "dPhi"],
        "hh_vec_DeltaEta": ["HH", "dEta"],
        "hh_CosThetaStar_CS": ["HH", "Costhetastar_CS"],
        "higgs1_tau2overtau3": ["HiggsLeading", "Tau3OverTau2"],
        "higgs2_tau2overtau3": ["HiggsSubLeading", "Tau3OverTau2"],
        "higgs1_reco_pt": ["HiggsLeading", "pt"],
        "higgs1_reco_eta": ["HiggsLeading", "eta"],
        "higgs1_reco_phi": ["HiggsLeading", "phi"],
        # "higgs1_reco_mass": ["HiggsLeading", "mass"],
        "higgs1_btag_wp": ["HiggsLeading", "btagBBTXbb_5wp"],  # nominal WP index of the leading Higgs (not rescaled, not merged; 1..5 after cut_sub_above_lead)
        # "higgs1_btag_morph": ["HiggsLeading", "btagBBTXbb_morph"],
        "higgs1_centrality": ["HiggsLeading", "centrality"],
        "higgs2_reco_pt": ["HiggsSubLeading", "pt"],
        "higgs2_reco_eta": ["HiggsSubLeading", "eta"],
        "higgs2_reco_phi": ["HiggsSubLeading", "phi"],
        "higgs2_reco_mass": ["HiggsSubLeading", "mass"],
        # "higgs2_btag_dig": ["HiggsSubLeading", "btagBBTXbb"],
        "higgs2_btag_cdf_wp": ["HiggsSubLeading", "btagBBTXbb_CDF_rescaled_WPs"],  # quantile-mapped, merged subleading WP index (1, 2, 3; txbb_cdf_max_wp = 3)
        "higgs2_centrality": ["HiggsSubLeading", "centrality"],
        "higgs1_divHHmass": ["HiggsLeading", "divHHmass"],
        "higgs1byhiggs2pt": ["events", "HiggsLeadingByHiggsSubLeadingPt"],
        "mjj": ["events", "mjjJetGoodVBFGood"],
        "deta": ["events", "detaJetGoodVBFGood"],
        "higgs1_dRclosestVBF": ["HiggsLeading", "dRclosestVBF"],
        "higgs2_dRclosestVBF": ["HiggsSubLeading", "dRclosestVBF"],
        "higgs1_massclosestVBF": ["HiggsLeading", "massclosestVBF"],
        "higgs2_massclosestVBF": ["HiggsSubLeading", "massclosestVBF"],
    }
)


test_set = OrderedDict(
    {
        "hh_vec_pt": ["HH", "pt"],
        "hh_vec_eta": ["HH", "eta"],
        "hh_vec_mass": ["HH", "mass"],
        "hh_vec_phi": ["HH", "phi"],
    }
)
