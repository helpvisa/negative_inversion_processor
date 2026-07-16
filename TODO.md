# TODO
- [ ] option to expose raw film density and skip luminance conversion
    - for theoretical use with film print LUTs
    - curious in general to know what this would look like
    - raw Linear Rec.2020 working image
- [ ] create a graphical user interface which includes:
    - **This whole thing is going to need a design document and some sketches**
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
