from ..var import PreprocessingType


def print_preprocessing_title(preprocessing_type: PreprocessingType):
    type_to_display = (
        preprocessing_type
        if preprocessing_type == "complete"
        else "  " + preprocessing_type + "  "
    )
    print("\n#########################################")
    print("#                                       #")
    print(f"#             PREPROCESSING   {type_to_display}  #")
    print("#                                       #")
    print("#########################################")


def print_predicting_title():
    print("\n#########################################")
    print("#                                       #")
    print("#            PREDICTING                 #")
    print("#                                       #")
    print("#########################################")


def print_job_done():
    print("\n#########################################")
    print("#                                       #")
    print("#               DONE                    #")
    print("#                                       #")
    print("#########################################")
