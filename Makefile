# lance-a-laya pipeline. Heavy steps run on Modal (see README); everything else is local Python.
PY := uv run python
MODAL := uv run modal
VOLUME := lal-data

.PHONY: setup test events sources audio upload smoke transcribe pull align windows teacher-packs labels \
	gold-sample gold zeroshot finetune calibrate predict bench review-pool review eval keyword-sweep demo clips plots all \
	report new-match

setup:            ## local env (light: no torch/laya locally)
	uv sync

test:
	uv run pytest -q

events:           ## StatsBomb -> data/events/*.jsonl
	$(PY) -m lal.events --include-spares

sources:          ## verify YouTube sources + caption coverage -> data/sources.json
	$(PY) -m lal.sources

audio:            ## download low-bitrate audio locally (data/audio/, gitignored)
	$(PY) -m lal.download

upload:           ## push local audio to the Modal Volume and delete local copies
	$(PY) -m lal.download --upload

smoke:            ## 10-17 min ASR smoke test on a training match (kickoff + 2 goals)
	$(MODAL) run -m lal.cloud.transcribe::main --match bra-kor-2022 --start 4400 --end 5400

transcribe:       ## transcribe all matches (clipped to match time) on Modal L4s, in parallel
	$(MODAL) run -m lal.cloud.transcribe::main

pull:             ## fetch transcripts from the Volume -> data/transcripts/ (gitignored)
	mkdir -p data/transcripts
	$(MODAL) volume get --force $(VOLUME) transcripts/ data/

align:            ## per-period clock->audio offsets + leave-one-out residuals -> data/align/*.json
	$(PY) -m lal.align

windows:          ## 20 s / 5 s sliding windows over in-play time -> data/interim/windows/ (gitignored)
	$(PY) -m lal.windows

teacher-packs:    ## transcript packs for the Claude teacher subagents (guide in src/lal/schema.py)
	$(PY) -m lal.teacher

labels:           ## merge event + teacher labels -> data/labels/, datasets/{train,val}.jsonl, balance report
	$(PY) -m lal.build_dataset

gold-sample:      ## freeze the stratified 150-window gold sample of the test match + audio snippets
	$(PY) -m lal.gold.sample

gold:             ## local labeling page on http://127.0.0.1:8765
	$(PY) -m lal.gold.server

zeroshot:         ## zero-shot laya-multilingual on val + test
	$(MODAL) run -m lal.cloud.predict::main --model zeroshot --splits val,test --tag zeroshot-en

RUN ?= ft-v1
finetune:         ## fine-tune on Modal (A100) -> Volume ckpt/$(RUN)
	$(MODAL) run -m lal.cloud.finetune::main --run $(RUN) --epochs 3

calibrate:        ## fit Laya temperatures on the val matches -> Volume calib/$(RUN).json
	$(MODAL) run -m lal.cloud.predict::calib --model $(RUN)

predict:          ## calibrated predictions on val + test
	$(MODAL) run -m lal.cloud.predict::main --model $(RUN) --calibration $(RUN) --splits val,test --tag $(RUN)-cal

bench:            ## latency on T4 + L4 -> outputs/latency_$(RUN).json
	$(MODAL) run -m lal.cloud.predict::bench --model $(RUN) --calibration $(RUN)

review-pool:      ## pooled big_chance/controversy triggers for the human review
	$(PY) -m lal.gold.review --ft $(RUN)-cal

review:           ## review page on http://127.0.0.1:8765
	$(PY) -m lal.gold.server --review

eval:             ## results on the gold set + event level -> outputs/results.json
	$(PY) -m lal.evaluate --ft $(RUN)-cal --ft-raw $(RUN)-T1

keyword-sweep:    ## keyword goal rule at 1/2/3 "gol" hits, event level -> outputs/keyword_sweep.json
	$(PY) -m lal.keyword_sweep

demo:             ## streaming replay of the test match -> outputs/timeline_*.json
	$(PY) -m lal.stream_demo --tag $(RUN)-cal

clips:            ## download only the highlight segments (local, gitignored)
	$(PY) -m lal.clips

plots:            ## figures -> outputs/*.png
	$(PY) -m lal.plots

report:           ## results tables (markdown) -> outputs/results.md
	$(PY) -m lal.report

MATCH ?= bra-col-2024
new-match:        ## inference only on a registered match without event data: audio -> ASR -> windows -> live stream
	$(PY) -m lal.download --match $(MATCH) --upload
	$(MODAL) run -m lal.cloud.transcribe::main --match $(MATCH) --include-spares
	$(MODAL) volume get --force $(VOLUME) transcripts/$(MATCH).jsonl data/transcripts/
	$(PY) -m lal.windows --match $(MATCH)
	$(PY) -m lal.stream_demo --match $(MATCH) --backend modal --tag $(RUN)-cal

all: events audio upload transcribe pull align windows teacher-packs labels finetune calibrate predict zeroshot bench eval demo plots
