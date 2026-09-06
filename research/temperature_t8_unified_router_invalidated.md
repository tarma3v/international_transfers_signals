# Temperature T8 invalidation

The first T8 build stopped on its timestamp-uniqueness assertion before writing
an artifact. It revealed that the frozen after-publication base is 18:30, while
the draft router had treated 18:30 as both the base and a later T7 refresh. The
draft also depended on the semantically invalid T7 experiment.

No T8 score table or reported metric exists. T8B uses the real 18:30 base and
only the corrected T7B 19:00/20:00 updates.
