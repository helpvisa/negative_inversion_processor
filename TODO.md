# TODO
## General
- option to expose raw film density and skip luminance conversion
    - for theoretical use with film print LUTs
    - curious in general to know what this would look like
    - raw Linear Rec.2020 working image

## CLI
- Validate output directory before attempting to process

## GUI
- folder batch loading / exporting / sidebar explorer + previewer
    - stretch goal: include sliders on each individual preview for ez grading
- revamp color picker
    - currently passes (x, y) tuple to GUI element, but instead:
        1. send signal with (x, y) tuple from ImageView
        2. connect above signal with processing pipeline to fetch (r, g, b)
           from that stage in the pipeline (i.e. pre-inversion, etc.)
        3. send signal containing fetched (r, g, b) from processing pipeline
        4. finally, connect above (r, g, b) signal with ColorPicker
        5. result: pipeline-accurate (r, g, b) color, no stale (x, y) values
- individual file tracking
    - track EditParams for each image that's been loaded
        - persist in-memory? sidecar? both?
    - copy/paste EditParams for easier batching
        - allow global paste in folder view
- tree view for loaded folders?

## NOTES
- how to apply the *exact same results* across entire roll in UI?
    - by default, median in each channel is used as reference
        - this is required for a decent inversion
    - sometimes, the median used can skew results across different images in
      a given roll
    - option to apply *exact same processed settings* to each image in roll?
        - circumvents UI edit params
        - possibly unintuitive?
    - "process entire roll with these settings" button?
