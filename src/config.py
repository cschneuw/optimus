class CorticalThicknessConfig:
    """CorticalThickness data parameters"""

    def __init__(self, **kwargs):
        super(CorticalThicknessConfig, self).__init__()

        self.file = "../../dataset/ADNI/ADNI_corticalThichness.csv"
        self.input_var="Schaefer_200_7"
        self.target_vars=["ADNI_MEM", "ADNI_EF", "ADNI_LAN", "ADNI_VS"]

        self.remove_medial_wall=True

class CorticalVolumeConfig: 
    """CorticalThickness data parameters"""
    def __init__(self, **kwargs):
        super(CorticalVolumeConfig, self).__init__()
        self.file = "../../../dataset/ADNI/MRI_PET_data.csv"
        self.input_var="MRI_Vol"
        self.target_vars=["ADNI_MEM", "ADNI_EF", "ADNI_LAN", "ADNI_VS"]

        self.remove_medial_wall=False

class PETActivityConfig: 
    """CorticalThickness data parameters"""
    def __init__(self, **kwargs):
        super(CorticalVolumeConfig, self).__init__()

        self.file = "../../../dataset/ADNI/MRI_PET_data.csv"
        self.input_var="PET_Vol"
        self.target_vars=["ADNI_MEM", "ADNI_EF", "ADNI_LAN", "ADNI_VS"]

        self.remove_medial_wall=False
