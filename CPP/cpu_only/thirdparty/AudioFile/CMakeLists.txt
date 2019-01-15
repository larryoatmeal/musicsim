set(LIB_NAME AudioFile)

set(CPP_FILES
    AudioFile.cpp
    )


set(CPP_HEADERS
    AudioFile.h
    )

add_library(${LIB_NAME} STATIC ${CPP_FILES} ${CPP_HEADERS})