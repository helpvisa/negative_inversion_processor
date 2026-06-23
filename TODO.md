# TODO
- [ ] include means of affecting `R_exp` and `B_exp`
    - picking two gray points
    - estimate using median + emulsion if no user input
- [ ] combine all multiplication effects to occur in single step
    - user `RGB` adjustments occur at same time as automated density balance
        - accumulate both adjustments, apply single time
    - apply shift in single step after all multiplication occurs in density
      space
        - same philosophy, accumulate auto-adjustment and user offset, apply
          once
- [ ] create a graphical user interface which includes:
    - [ ] folder batch loading / exporting
    - [ ] roll preview
        - stretch goal: include sliders on each individual preview in roll
          preview mode, similar to professional scanning software
    - [ ] image preview
    - [ ] `RGB` adjustment knobs
    - [ ] white balance offset sliders
    - [ ] white point picker
    - [ ] `OCIO` options configurator
    - [ ] image processing on non-GUI thread (avoid blocking)
- [ ] included `OCIO` config which can convert from internal Linear Rec.2020
      colourspace to user-chosen "input" colourspace
    - combine with user-provided `OCIO` configuration for flexibility
    - allows for application of `LUTs` in any of the user's preferred colour
      spaces
    - allows for export to any colourspace
    - still can't enable colourspace tagging by default (`OCIO` can't bake
      `ICC` profiles)
