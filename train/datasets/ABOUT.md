Dataset 1

There are around 1500 unique patients, of this set 391 had no extracted label, 272 had one diagnosis label, 290 had two,
and 534 had more than 2 (see img 1 for breakdown). There were a few outliers that had >10 labels, which could be the
result of errors during label extraction, error during clustering, or just due to patients with a very high number of
disorders.

out of the 8k files we id'd as clinically relevant, the majority actually include 3 or more speakers (60%). Looking more
closely, there's a lot of cases in which there are two 'main' speakers who take the majority of turns, and one or more
speakers with only a handful of utterences (see img. 3). so, we may be able to do some filtering based on that (open for
discussion).

regarding the number of cases in which a transcription identified more than 2 people (i.e. more than just the doctor and
the patient): from all the files that Bruno clustered, about 30% of transcriptions seem to have more than two speakers.
I'll have to do more digging to see if that percentage changes when looking at just the transcriptions for sessions id'd
as clinical