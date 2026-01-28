file(REMOVE_RECURSE
  "libacd_.a"
  "libacd_.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/acd_.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
