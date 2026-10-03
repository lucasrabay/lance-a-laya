"""The single label schema: Laya questions + labeling guide. Training, evaluation, the teacher
labelers, the gold labeling page and the demo all import from here.

Notes from the Laya 0.3.24 docs that shaped this file:
- every `noul` carries explicit `criteria` keyed `true`/`false` (a criteria-less noul is unreliable);
- no `labels` override: the fine-tuning script never renders it, so inference must not either;
- `score` is the weakest primitive and laya-multilingual rarely picks the first-listed level (#131),
  which is a known handicap for `intensity` zero-shot (most windows are "calm").
"""

from __future__ import annotations

NOUL_QUESTIONS = ("goal", "big_chance", "controversy", "card")
SCORE_QUESTIONS = ("intensity",)
QUESTION_IDS = NOUL_QUESTIONS + SCORE_QUESTIONS
INTENSITY_LEVELS = ("calm", "building", "peak")

QUESTIONS: dict[str, dict] = {
    "goal": {
        "type": "noul",
        "instructions": "Football narration window. Is a goal scored live in this window?",
        "criteria": {
            "true": "the narrator calls a goal happening now: the ball goes in and the goal is celebrated",
            "false": "no goal is scored now: attacks, misses, saves, disallowed goals, replays or talk "
                     "about earlier goals",
        },
    },
    "big_chance": {
        "type": "noul",
        "instructions": "Football narration window. Is there a clear scoring chance that does not "
                        "end in a goal?",
        "criteria": {
            "true": "a dangerous shot or header is saved, hits the post or bar, or narrowly misses, "
                    "and the narrator reacts to the near miss",
            "false": "no clear chance: build-up play, midfield, fouls, stoppages, or a goal that was "
                     "actually scored",
        },
    },
    "controversy": {
        "type": "noul",
        "instructions": "Football narration window. Is a refereeing decision being contested or reviewed?",
        "criteria": {
            "true": "VAR review, a disputed penalty, offside or foul, players or narrators protesting "
                    "the referee, or a goal being disallowed",
            "false": "play goes on without dispute: routine fouls, throw-ins, free kicks and decisions "
                     "nobody protests",
        },
    },
    "card": {
        "type": "noul",
        "instructions": "Football narration window. Is a yellow or red card shown in this window?",
        "criteria": {
            "true": "the narrator says a player is booked or sent off right now (yellow card, red card)",
            "false": "no card is shown now: fouls without a card, or talk about cards shown earlier",
        },
    },
    "intensity": {
        "type": "score",
        "instructions": "Football narration window. How intense is the moment being narrated?",
        "criteria": [
            "calm: possession in midfield, slow build-up, chatter, stoppages or replays",
            "building: an attack develops, the ball nears the box, crosses, corners or free kicks",
            "peak: a shot, save, goal or decisive moment with the narrator shouting",
        ],
    },
}

# Same questions with PT-BR instructions/criteria, for the zero-shot instruction-language comparison on the
# validation matches. Keys and option order are identical, so labels and metrics carry over.
QUESTIONS_PT: dict[str, dict] = {
    "goal": {
        "type": "noul",
        "instructions": "Trecho de narração de futebol. Sai um gol ao vivo neste trecho?",
        "criteria": {
            "true": "o narrador grita um gol que acontece agora: a bola entra e o gol é comemorado",
            "false": "não sai gol agora: ataques, chutes para fora, defesas, gol anulado, replay ou "
                     "comentário sobre gols anteriores",
        },
    },
    "big_chance": {
        "type": "noul",
        "instructions": "Trecho de narração de futebol. Há uma chance clara de gol que não termina em gol?",
        "criteria": {
            "true": "um chute ou cabeceio perigoso é defendido, bate na trave ou passa raspando, e o "
                    "narrador reage ao quase gol",
            "false": "nenhuma chance clara: troca de passes, meio-campo, faltas, paralisações ou um gol "
                     "que de fato saiu",
        },
    },
    "controversy": {
        "type": "noul",
        "instructions": "Trecho de narração de futebol. Uma decisão da arbitragem está sendo contestada "
                        "ou revisada?",
        "criteria": {
            "true": "revisão do VAR, pênalti, impedimento ou falta polêmica, reclamação contra o árbitro, "
                    "ou um gol sendo anulado",
            "false": "o jogo segue sem polêmica: faltas comuns, laterais, tiros livres e decisões que "
                     "ninguém contesta",
        },
    },
    "card": {
        "type": "noul",
        "instructions": "Trecho de narração de futebol. Um cartão amarelo ou vermelho é mostrado neste trecho?",
        "criteria": {
            "true": "o narrador diz que um jogador leva cartão ou é expulso agora (cartão amarelo, "
                    "cartão vermelho)",
            "false": "nenhum cartão agora: faltas sem cartão, ou comentário sobre cartões anteriores",
        },
    },
    "intensity": {
        "type": "score",
        "instructions": "Trecho de narração de futebol. Qual a intensidade do momento narrado?",
        "criteria": [
            "calmo: posse no meio-campo, construção lenta, conversa, paralisação ou replay",
            "crescendo: um ataque se desenha, a bola chega perto da área, cruzamentos, escanteios ou faltas",
            "pico: chute, defesa, gol ou lance decisivo com o narrador gritando",
        ],
    },
}


def questions_for(variant: str = "en") -> dict[str, dict]:
    return {"en": QUESTIONS, "pt": QUESTIONS_PT}[variant]


# Plain-language guide shared by the teacher labelers, the gold labeling page and the README.
LABELING_GUIDE = {
    "goal": "Positive only for the live call of a goal and its immediate celebration. Replays, studio "
            "recaps and 'remember the goal' talk are negative. A goal disallowed by VAR is NOT a goal "
            "(it is controversy).",
    "big_chance": "A clear chance to score that does not go in: a save from close range, a shot off the "
                  "post/bar, a miss from a dangerous position, a one-on-one. Speculative long shots and "
                  "routine crosses are negative.",
    "controversy": "A contested or reviewed refereeing decision: VAR checks, penalty or offside disputes, "
                   "protests, disallowed goals, a narrator questioning the referee. A routine foul or a "
                   "card nobody disputes is negative.",
    "card": "Positive where the narration announces a yellow or red card being shown. The foul itself, "
            "before the referee decides, is negative; later mentions of an earlier card are negative.",
    "intensity": "calm = nothing threatening; building = an attack, set piece or pressure near the box; "
                 "peak = shot, save, goal or a shouting narrator at a decisive moment.",
}


def state_for(text: str, prev_text: str | None = None) -> dict:
    """The Laya state for a window. `prev_text` (previous window) is an optional ablation."""
    if prev_text is None:
        return {"narracao": text}
    return {"narracao_anterior": prev_text, "narracao": text}


def validate() -> None:
    assert list(QUESTIONS) == list(QUESTIONS_PT)
    for qid, q in list(QUESTIONS.items()) + list(QUESTIONS_PT.items()):
        if q["type"] == "noul":
            assert set(q["criteria"]) == {"true", "false"}, qid
            assert "labels" not in q, qid
        elif q["type"] == "score":
            assert isinstance(q["criteria"], list) and len(q["criteria"]) == len(INTENSITY_LEVELS), qid


validate()
