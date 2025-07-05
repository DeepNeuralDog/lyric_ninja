import coremltools.models.utils as coreml_utils

if __name__ == "__main__":
    mlpackage_path = "/Users/hasibulhasan/github/lyric_ninja/models/wav2_vec2.mlpackage"
    output_path = "/Users/hasibulhasan/github/lyric_ninja/models/wav2_vec2.mlmodelc"
    coreml_utils.compile_model(model=mlpackage_path, destination_path=output_path)
