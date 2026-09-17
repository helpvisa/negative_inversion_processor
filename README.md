# Negative Inversion Processor
## Or: NIP

'Negative Inversion Processor' (or NIP, for short) helps you accurately and
efficiently invert your photographic negatives to generate lossless digital
intermediates *or* fully-finished and cropped images using ACES tonemapping.

'NIP' was born of necessity: I strongly felt that existing options for
open-source negative inversion, though often offering more advanced UIs and
expanded feature-sets, lacked transparency when it came to the process used for
actually inverting your negatives. The goals were simple:

- Accurately invert negatives and correct unwanted colour casts
- Offer *optional* post-inversion grading and tonemapping
- Control over the final export and colour profile
- Apply inversion parameters to an entire roll and allow batch exporting

Most of my editing occurs in external software like
[`darktable`](https://www.darktable.org/), and existing options didn't make it
easy to export inversions in a known colour space with no tonemapping. NIP
takes inspiration from ["Cineon"](https://en.wikipedia.org/wiki/Cineon) in this
fashion: generate your digital intermediates with absolute no processing beyond
the required inversion and colour correction, then edit them elsewhere. I
strongly believe this offers the best possible results for digital scanning and
finishing of film photography.

The code shown in this repository constitutes an alpha release, and
documentation on the features of NIP (as well as how to use it) is not yet
complete. If you would like to try it for yourself, you can follow the steps
outlined below:

1. Clone this repository using `git`
2. Create a virtual python environment in the cloned repository and activate it
3. Install the necessary dependencies using pip (`pip install -r
   requirements.txt`)
4. Run NIP in UI mode using `python3 ./inverter/ui.py`

Included in this repository are also some shell scripts which demonstrate how
the CLI component of NIP can be used for non-UI batch processing of
negatives. Please be forewarned that the NIP CLI is not maintained at the same
pace of the UI, and may not possess the full set of features exposed via the
UI.

The code in this repository is distributed under the terms of the GPLv3 , a
copy of which should be present and included here in [`LICENSE`](LICENSE). This
repository also contains some tonemapping functions which use code found in
other MIT-licensed projects; their respective licenses as well as the functions
used can be found in [`THIRD_PARTY_LICENSES.txt`](THIRD_PARTY_LICENSES.txt).
