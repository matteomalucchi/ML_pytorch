from collections import OrderedDict

bkg_morphing_dnn_input_variables = OrderedDict(
    {
        "era": ["events", "era"],
        "higgs1_reco_pt": ["HiggsLeading", "pt"],
        "higgs1_reco_eta": ["HiggsLeading", "eta"],
        "higgs1_reco_phi": ["HiggsLeading", "phi"],
        "higgs1_reco_mass": ["HiggsLeading", "mass"],
        "higgs2_reco_pt": ["HiggsSubLeading", "pt"],
        "higgs2_reco_eta": ["HiggsSubLeading", "eta"],
        "higgs2_reco_phi": ["HiggsSubLeading", "phi"],
        "higgs2_reco_mass": ["HiggsSubLeading", "mass"],
        "HT": ["events", "HT"],
        "higgs1_DeltaRjj": ["HiggsLeading", "dR"],
        "higgs2_DeltaRjj": ["HiggsSubLeading", "dR"],
        "minDeltaR_Higgjj": ["events", "dR_min"],
        "maxDeltaR_Higgjj": ["events", "dR_max"],
        "higgs1_helicityCosTheta": ["HiggsLeading", "helicityCosTheta"],
        "higgs2_helicityCosTheta": ["HiggsSubLeading", "helicityCosTheta"],
        "hh_CosThetaStar_CS": ["HH", "Costhetastar_CS"],
        "hh_vec_mass": ["HH", "mass"],
        "hh_vec_pt": ["HH", "pt"],
        "hh_vec_eta": ["HH", "eta"],
        "hh_vec_DeltaR": ["HH", "dR"],
        "hh_vec_DeltaPhi": ["HH", "dPhi"],
        "hh_vec_DeltaEta": ["HH", "dEta"],
        "higgs1_reco_jet1_pt": ["JetGoodFromHiggsOrdered:0", "pt"],
        "higgs1_reco_jet1_eta": ["JetGoodFromHiggsOrdered:0", "eta"],
        "higgs1_reco_jet1_phi": ["JetGoodFromHiggsOrdered:0", "phi"],
        "higgs1_reco_jet1_mass": ["JetGoodFromHiggsOrdered:0", "mass"],
        "higgs1_reco_jet2_pt": ["JetGoodFromHiggsOrdered:1", "pt"],
        "higgs1_reco_jet2_eta": ["JetGoodFromHiggsOrdered:1", "eta"],
        "higgs1_reco_jet2_phi": ["JetGoodFromHiggsOrdered:1", "phi"],
        "higgs1_reco_jet2_mass": ["JetGoodFromHiggsOrdered:1", "mass"],
        "higgs2_reco_jet1_pt": ["JetGoodFromHiggsOrdered:2", "pt"],
        "higgs2_reco_jet1_eta": ["JetGoodFromHiggsOrdered:2", "eta"],
        "higgs2_reco_jet1_phi": ["JetGoodFromHiggsOrdered:2", "phi"],
        "higgs2_reco_jet1_mass": ["JetGoodFromHiggsOrdered:2", "mass"],
        "higgs2_reco_jet2_pt": ["JetGoodFromHiggsOrdered:3", "pt"],
        "higgs2_reco_jet2_eta": ["JetGoodFromHiggsOrdered:3", "eta"],
        "higgs2_reco_jet2_phi": ["JetGoodFromHiggsOrdered:3", "phi"],
        "higgs2_reco_jet2_mass": ["JetGoodFromHiggsOrdered:3", "mass"],
        "add_jet1pt_pt": ["add_jet1pt", "pt"],
        "add_jet1pt_eta": ["add_jet1pt", "eta"],
        "add_jet1pt_phi": ["add_jet1pt", "phi"],
        "add_jet1pt_mass": ["add_jet1pt", "mass"],
        "sigma_over_higgs1_reco_mass": ["events", "sigma_over_higgs1_reco_mass"],
        "sigma_over_higgs2_reco_mass": ["events", "sigma_over_higgs2_reco_mass"],
    }
)

set_with_btag = OrderedDict(
    {
        "era": ["events", "era"],
        "higgs1_reco_pt": ["HiggsLeading", "pt"],
        "higgs1_reco_eta": ["HiggsLeading", "eta"],
        "higgs1_reco_phi": ["HiggsLeading", "phi"],
        "higgs1_reco_mass": ["HiggsLeading", "mass"],
        "higgs2_reco_pt": ["HiggsSubLeading", "pt"],
        "higgs2_reco_eta": ["HiggsSubLeading", "eta"],
        "higgs2_reco_phi": ["HiggsSubLeading", "phi"],
        "higgs2_reco_mass": ["HiggsSubLeading", "mass"],
        "HT": ["events", "HT"],
        "higgs1_DeltaRjj": ["HiggsLeading", "dR"],
        "higgs2_DeltaRjj": ["HiggsSubLeading", "dR"],
        "minDeltaR_Higgjj": ["events", "dR_min"],
        "maxDeltaR_Higgjj": ["events", "dR_max"],
        "higgs1_helicityCosTheta": ["HiggsLeading", "helicityCosTheta"],
        "higgs2_helicityCosTheta": ["HiggsSubLeading", "helicityCosTheta"],
        "hh_CosThetaStar_CS": ["HH", "Costhetastar_CS"],
        "hh_vec_mass": ["HH", "mass"],
        "hh_vec_pt": ["HH", "pt"],
        "hh_vec_eta": ["HH", "eta"],
        "hh_vec_DeltaR": ["HH", "dR"],
        "hh_vec_DeltaPhi": ["HH", "dPhi"],
        "hh_vec_DeltaEta": ["HH", "dEta"],
        "higgs1_reco_jet1_pt": ["JetGoodFromHiggsOrdered:0", "pt"],
        "higgs1_reco_jet1_eta": ["JetGoodFromHiggsOrdered:0", "eta"],
        "higgs1_reco_jet1_phi": ["JetGoodFromHiggsOrdered:0", "phi"],
        "higgs1_reco_jet1_mass": ["JetGoodFromHiggsOrdered:0", "mass"],
        #"higgs1_reco_jet1_btag": ["JetGoodFromHiggsOrdered:0", "btagPNetB"],
        "higgs1_reco_jet2_pt": ["JetGoodFromHiggsOrdered:1", "pt"],
        "higgs1_reco_jet2_eta": ["JetGoodFromHiggsOrdered:1", "eta"],
        "higgs1_reco_jet2_phi": ["JetGoodFromHiggsOrdered:1", "phi"],
        "higgs1_reco_jet2_mass": ["JetGoodFromHiggsOrdered:1", "mass"],
        #"higgs1_reco_jet2_btag": ["JetGoodFromHiggsOrdered:0", "btagPNetB"],
        "higgs2_reco_jet1_pt": ["JetGoodFromHiggsOrdered:2", "pt"],
        "higgs2_reco_jet1_eta": ["JetGoodFromHiggsOrdered:2", "eta"],
        "higgs2_reco_jet1_phi": ["JetGoodFromHiggsOrdered:2", "phi"],
        "higgs2_reco_jet1_mass": ["JetGoodFromHiggsOrdered:2", "mass"],
        #"higgs2_reco_jet1_btag": ["JetGoodFromHiggsOrdered:0", "btagPNetB"],
        "higgs2_reco_jet2_pt": ["JetGoodFromHiggsOrdered:3", "pt"],
        "higgs2_reco_jet2_eta": ["JetGoodFromHiggsOrdered:3", "eta"],
        "higgs2_reco_jet2_phi": ["JetGoodFromHiggsOrdered:3", "phi"],
        "higgs2_reco_jet2_mass": ["JetGoodFromHiggsOrdered:3", "mass"],
        #"higgs2_reco_jet2_btag": ["JetGoodFromHiggsOrdered:0", "btagPNetB"],
        "add_jet1pt_pt": ["add_jet1pt", "pt"],
        "add_jet1pt_eta": ["add_jet1pt", "eta"],
        "add_jet1pt_phi": ["add_jet1pt", "phi"],
        "add_jet1pt_mass": ["add_jet1pt", "mass"],
        "sigma_over_higgs1_reco_mass": ["events", "sigma_over_higgs1_reco_mass"],
        "sigma_over_higgs2_reco_mass": ["events", "sigma_over_higgs2_reco_mass"],
    }
)


test_set = OrderedDict(
    {
        "higgs1_reco_pt": ["HiggsLeading", "pt"],
        "higgs1_reco_eta": ["HiggsLeading", "eta"],
        "higgs1_reco_phi": ["HiggsLeading", "phi"],
        "higgs1_reco_mass": ["HiggsLeading", "mass"],
        "higgs2_reco_pt": ["HiggsSubLeading", "pt"],
        "higgs2_reco_eta": ["HiggsSubLeading", "eta"],
        "higgs2_reco_phi": ["HiggsSubLeading", "phi"],
        "higgs2_reco_mass": ["HiggsSubLeading", "mass"],
        "higgs1_DeltaRjj": ["HiggsLeading", "dR"],
        "higgs2_DeltaRjj": ["HiggsSubLeading", "dR"],
        "higgs1_reco_jet1_pt": ["JetGoodFromHiggsOrdered:0", "pt"],
        "higgs1_reco_jet1_eta": ["JetGoodFromHiggsOrdered:0", "eta"],
        "higgs1_reco_jet1_phi": ["JetGoodFromHiggsOrdered:0", "phi"],
        "higgs1_reco_jet1_mass": ["JetGoodFromHiggsOrdered:0", "mass"],
        "higgs1_reco_jet2_pt": ["JetGoodFromHiggsOrdered:1", "pt"],
        "higgs1_reco_jet2_eta": ["JetGoodFromHiggsOrdered:1", "eta"],
        "higgs1_reco_jet2_phi": ["JetGoodFromHiggsOrdered:1", "phi"],
        "higgs1_reco_jet2_mass": ["JetGoodFromHiggsOrdered:1", "mass"],
        "higgs2_reco_jet1_pt": ["JetGoodFromHiggsOrdered:2", "pt"],
        "higgs2_reco_jet1_eta": ["JetGoodFromHiggsOrdered:2", "eta"],
        "higgs2_reco_jet1_phi": ["JetGoodFromHiggsOrdered:2", "phi"],
        "higgs2_reco_jet1_mass": ["JetGoodFromHiggsOrdered:2", "mass"],
        "higgs2_reco_jet2_pt": ["JetGoodFromHiggsOrdered:3", "pt"],
        "higgs2_reco_jet2_eta": ["JetGoodFromHiggsOrdered:3", "eta"],
        "higgs2_reco_jet2_phi": ["JetGoodFromHiggsOrdered:3", "phi"],
        "higgs2_reco_jet2_mass": ["JetGoodFromHiggsOrdered:3", "mass"],
        "add_jet1pt_pt": ["add_jet1pt", "pt"],
        "add_jet1pt_eta": ["add_jet1pt", "eta"],
        "add_jet1pt_phi": ["add_jet1pt", "phi"],
        "add_jet1pt_mass": ["add_jet1pt", "mass"],
    }
)


sig_bkg_dnn_input_variables = OrderedDict(
    [
        ("era", ["events", "era"]),
        ("HT", ["events", "HT"]),
        ("hh_vec_mass", ["HH", "mass"]),
        ("hh_vec_pt", ["HH", "pt"]),
        ("hh_vec_eta", ["HH", "eta"]),
        ("hh_vec_phi", ["HH", "phi"]),
        ("hh_vec_DeltaPhi", ["HH", "dPhi"]),
        ("hh_vec_DeltaEta", ["HH", "dEta"]),
        ("hh_vec_DeltaR", ["HH", "dR"]),
        ("hh_CosThetaStar_CS", ["HH", "Costhetastar_CS"]),
        ("higgs1_reco_pt", ["HiggsLeading", "pt"]),
        ("higgs1_reco_eta", ["HiggsLeading", "eta"]),
        ("higgs1_reco_phi", ["HiggsLeading", "phi"]),
        ("higgs1_reco_mass", ["HiggsLeading", "mass"]),
        ("higgs2_reco_pt", ["HiggsSubLeading", "pt"]),
        ("higgs2_reco_eta", ["HiggsSubLeading", "eta"]),
        ("higgs2_reco_phi", ["HiggsSubLeading", "phi"]),
        ("higgs2_reco_mass", ["HiggsSubLeading", "mass"]),
        ("higgs1_DeltaPhijj", ["HiggsLeading", "dPhi"]),
        ("higgs2_DeltaPhijj", ["HiggsSubLeading", "dPhi"]),
        ("higgs1_DeltaEtajj", ["HiggsLeading", "dEta"]),
        ("higgs2_DeltaEtajj", ["HiggsSubLeading", "dEta"]),
        ("minDeltaR_Higgjj", ["events", "dR_min"]),
        ("maxDeltaR_Higgjj", ["events", "dR_max"]),
        ("higgs1_helicityCosTheta", ["HiggsLeading", "helicityCosTheta"]),
        ("higgs2_helicityCosTheta", ["HiggsSubLeading", "helicityCosTheta"]),
        ("higgs1_reco_jet1_pt", ["JetGoodFromHiggsOrdered:0", "pt"]),
        ("higgs1_reco_jet1_eta", ["JetGoodFromHiggsOrdered:0", "eta"]),
        ("higgs1_reco_jet1_phi", ["JetGoodFromHiggsOrdered:0", "phi"]),
        ("higgs1_reco_jet1_mass", ["JetGoodFromHiggsOrdered:0", "mass"]),
        ("higgs1_reco_jet2_pt", ["JetGoodFromHiggsOrdered:1", "pt"]),
        ("higgs1_reco_jet2_eta", ["JetGoodFromHiggsOrdered:1", "eta"]),
        ("higgs1_reco_jet2_phi", ["JetGoodFromHiggsOrdered:1", "phi"]),
        ("higgs1_reco_jet2_mass", ["JetGoodFromHiggsOrdered:1", "mass"]),
        ("higgs2_reco_jet1_pt", ["JetGoodFromHiggsOrdered:2", "pt"]),
        ("higgs2_reco_jet1_eta", ["JetGoodFromHiggsOrdered:2", "eta"]),
        ("higgs2_reco_jet1_phi", ["JetGoodFromHiggsOrdered:2", "phi"]),
        ("higgs2_reco_jet1_mass", ["JetGoodFromHiggsOrdered:2", "mass"]),
        ("higgs2_reco_jet2_pt", ["JetGoodFromHiggsOrdered:3", "pt"]),
        ("higgs2_reco_jet2_eta", ["JetGoodFromHiggsOrdered:3", "eta"]),
        ("higgs2_reco_jet2_phi", ["JetGoodFromHiggsOrdered:3", "phi"]),
        ("higgs2_reco_jet2_mass", ["JetGoodFromHiggsOrdered:3", "mass"]),
        ("add_jet1pt_pt", ["add_jet1pt", "pt"]),
        ("add_jet1pt_eta", ["add_jet1pt", "eta"]),
        ("add_jet1pt_phi", ["add_jet1pt", "phi"]),
        ("add_jet1pt_mass", ["add_jet1pt", "mass"]),
        ("add_jet1pt_Higgs1_deta", ["add_jet1pt", "LeadingHiggs_dEta"]),
        ("add_jet1pt_Higgs1_dphi", ["add_jet1pt", "LeadingHiggs_dPhi"]),
        ("add_jet1pt_Higgs1_m", ["add_jet1pt", "LeadingHiggs_mass"]),
        ("add_jet1pt_Higgs2_deta", ["add_jet1pt", "SubLeadingHiggs_dEta"]),
        ("add_jet1pt_Higgs2_dphi", ["add_jet1pt", "SubLeadingHiggs_dPhi"]),
        ("add_jet1pt_Higgs2_m", ["add_jet1pt", "SubLeadingHiggs_mass"]),
        ("sigma_over_higgs1_reco_mass", ["events", "sigma_over_higgs1_reco_mass"]),
        ("sigma_over_higgs2_reco_mass", ["events", "sigma_over_higgs2_reco_mass"]),
    ]
)
