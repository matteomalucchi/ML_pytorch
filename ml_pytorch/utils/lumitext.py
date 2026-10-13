"""Build the luminosity text of the plots from the datasets used in the training."""

import os

# data taking periods that can appear in the dataset names, grouped by year
YEAR_PERIODS = {
    "2022": ["2022_preEE", "2022_postEE"],
    "2023": ["2023_preBPix", "2023_postBPix"],
    "2024": ["2024"],
}

ENERGY_TEXT = "(13.6 TeV)"


def get_years_text(datasets):
    """Get the years from a list of dataset names.

    If all the periods of a year appear, only the year is returned (e.g. "2022"),
    otherwise the period itself (e.g. "2022_postEE").
    Multiple years are summed up (e.g. "2022 + 2023").
    """
    datasets = [str(dataset) for dataset in datasets]
    years = []
    for year, periods in YEAR_PERIODS.items():
        found = [
            period
            for period in periods
            if any(period in dataset for dataset in datasets)
        ]
        if not found:
            continue
        if len(found) == len(periods):
            years.append(year)
        else:
            years += found
    return " + ".join(years)


def get_lumitext(cfg=None):
    """Get the luminosity text of the plots from the signal and background
    datasets defined in the config."""
    if cfg is None:
        return ENERGY_TEXT

    datasets = []
    for key in ["signal_dataset", "background_dataset"]:
        value = cfg.get(key, None)
        if value is None:
            continue
        if isinstance(value, str):
            datasets.append(value)
        else:
            datasets += list(value)

    years_text = get_years_text(datasets)
    return f"{years_text} {ENERGY_TEXT}" if years_text else ENERGY_TEXT


def get_lumitext_from_dir(dir):
    """Get the luminosity text from the config saved in the output directory
    of the training (`config_parameters.yml`)."""
    cfg_file = os.path.join(dir, "config_parameters.yml")
    if not os.path.exists(cfg_file):
        return ENERGY_TEXT

    from omegaconf import OmegaConf

    return get_lumitext(OmegaConf.load(cfg_file))
