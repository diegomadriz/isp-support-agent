import argparse
import asyncio
import json
from pathlib import Path

from .adapters import scenarios
from .classification import rules
from .demo import run_demo
from .evaluation import evaluate_suite, replay, write_results
from .language_models import make_model
from .logging_config import configure_logging
from .recordings import gate_graph, record_blind, record_suite, replay_graph, write_graph_results
from .runtime import AgentRuntime
from .settings import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="ISP Support Agent · simulated network")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Replay all conversations offline")
    demo.add_argument("--output", type=Path, default=Path("docs"))
    chat = sub.add_parser("chat")
    chat.add_argument("--scenario", choices=list(scenarios()))
    serve = sub.add_parser("serve")
    serve.add_argument("--scenario", choices=list(scenarios()))
    evaluation = sub.add_parser(
        "eval", help="Score development, synthetic test and any recorded blind messages"
    )
    evaluation.add_argument(
        "--mode", choices=["rules-only", "rules+llm", "suite"], default="rules-only"
    )
    evaluation.add_argument("--directory", type=Path, default=Path("eval"))
    evaluation.add_argument("--output", type=Path)
    saved = evaluation.add_mutually_exclusive_group()
    saved.add_argument(
        "--replay",
        type=Path,
        help="Regenerate metrics from saved inference, without calling a model",
    )
    evaluation.add_argument(
        "--calibrate", action="store_true", help="Choose threshold on dev only before running test"
    )
    evaluation.add_argument("--gate", action="store_true")
    saved.add_argument("--record", type=Path, help="Capture raw live responses and fingerprints")
    saved.add_argument(
        "--record-blind",
        action="store_true",
        help="Record new blind messages with the pinned live model and add them to the gate",
    )
    saved.add_argument(
        "--graph-replay", type=Path, help="Replay raw responses through the compiled graph"
    )
    example = sub.add_parser(
        "example", help="Record a configured-model conversation beside the reference rule"
    )
    example.add_argument("--output", type=Path, default=Path("docs/model-example.md"))
    diagram = sub.add_parser("diagram")
    diagram.add_argument("--output", type=Path, default=Path("docs/graph.mmd"))
    rewrite_eval = sub.add_parser(
        "rewrite-eval", help="Record or exactly replay paired reply rewriting"
    )
    rewrite_mode = rewrite_eval.add_mutually_exclusive_group(required=True)
    rewrite_mode.add_argument("--record", type=Path)
    rewrite_mode.add_argument("--replay", type=Path)
    rewrite_eval.add_argument("--output", type=Path, default=Path("docs"))
    rewrite_eval.add_argument("--gate", action="store_true")
    rewrite_eval.add_argument(
        "--answer-key", type=Path, help="Write review key outside the public repo"
    )
    args = parser.parse_args(argv)
    if args.command == "rewrite-eval":
        from .rewrite_experiment import (
            gate,
            record_experiment,
            replay_experiment,
        )
        from .rewrite_experiment import (
            write_results as write_rewrite_results,
        )
        from .rewrite_experiment import (
            write_review as write_rewrite_review,
        )

        if args.record:
            bundle, result = record_experiment(Path.cwd(), Settings.from_env(), args.record)
        else:
            bundle = json.loads(args.replay.read_text())
            result = replay_experiment(bundle, Path.cwd())
        if args.gate:
            gate(result)
        write_rewrite_results(result, args.output)
        if args.answer_key:
            write_rewrite_review(bundle, args.answer_key.parent, args.answer_key)
        print(
            f"Reply acceptance {result['acceptance_rate']:.1%}; leaks {result['leaks']}; "
            f"facts {result['required_facts_preserved']:.1%}; p95 added latency "
            f"{result['added_latency_ms']['p95']} ms; {result['decision']}"
        )
        return
    if args.command == "demo":
        print(f"Replayed {len(run_demo(args.output))} offline scenarios in {args.output}")
        return
    if args.command == "eval":
        if (args.record or args.record_blind or args.graph_replay) and args.directory != Path(
            "eval"
        ):
            parser.error("Raw recording/replay uses the frozen eval/ datasets")
        if args.record_blind:
            if args.calibrate or args.gate:
                parser.error(
                    "Blind recording uses frozen settings and runs the replay gate automatically"
                )
            asyncio.run(
                record_blind(
                    Path.cwd(),
                    Settings.from_env(),
                    Path("eval/recordings.json"),
                    args.output or Path("docs/eval-ollama.json"),
                )
            )
            return
        if args.graph_replay:
            if args.calibrate:
                parser.error("Recorded replay cannot recalibrate frozen settings")
            bundle = json.loads(args.graph_replay.read_text())
            result = asyncio.run(replay_graph(bundle, Path.cwd()))
            if args.gate:
                gate_graph(result, bundle["baseline"])
            write_graph_results(result, args.output or Path("docs/eval-replay.json"))
            for split, r in result["splits"].items():
                print(
                    f"{split}/graph: accuracy {r['accuracy']:.1%}; critical recall {r['critical_recall']:.1%}"
                    if r.get("critical_count", 1)
                    else f"{split}/graph: accuracy {r['accuracy']:.1%}; no critical rows"
                )
            return
        if args.gate:
            parser.error(
                "Use --graph-replay eval/recordings.json --gate; rules/stub is not the safety gate"
            )
        if args.record and (args.replay or args.calibrate or args.mode != "suite"):
            parser.error(
                "--record requires --mode suite, without --replay or --calibrate; settings are frozen"
            )
        settings = Settings() if args.mode == "rules-only" else Settings.from_env()
        if args.record:
            result = asyncio.run(
                record_suite(
                    Path.cwd(), settings, args.record, args.output or Path("docs/eval-ollama.json")
                )
            )
        elif args.replay:
            result = replay(json.loads(args.replay.read_text()))
        else:

            async def run():
                model = make_model(settings)
                try:
                    return await evaluate_suite(
                        args.directory, model, settings, calibrate=args.calibrate
                    )
                finally:
                    await model.close()

            result = asyncio.run(run())
        backend = result["metadata"]["backend"]
        output = args.output or Path(f"docs/eval-{'rules' if backend == 'stub' else backend}.json")
        if result["metadata"]["backend"] == "stub":
            # Report the reference rules, never learned-model accuracy for a stub.
            result["dev_calibration"] = []
            result["metadata"]["benchmark_inferences"] = 0
            for split in result["splits"].values():
                split["modes"] = {"rules-only": split["modes"]["rules-only"]}
        write_results(result, output)
        for split, data in result["splits"].items():
            for mode, r in data["modes"].items():
                print(
                    (
                        f"{split}/{mode}: accuracy {r['accuracy']:.1%}; critical recall {r['critical_recall']:.1%}"
                        if r.get("critical_count", 1)
                        else f"{split}/{mode}: accuracy {r['accuracy']:.1%}; no critical rows"
                    )
                )
        return
    settings = Settings() if args.command == "diagram" else Settings.from_env()
    if getattr(args, "scenario", None):
        settings = settings.model_copy(update={"scenario": args.scenario})
    if args.command == "serve":
        from .web import create_app

        configure_logging()
        runtime = AgentRuntime.persistent(settings.model_copy(update={"staff_panel": True}))
        app = create_app(runtime)
        try:
            app.run(host=settings.host, port=settings.port, debug=False, use_reloader=False)
        finally:
            app.extensions["background"].close()
            runtime.close()
        return
    if args.command == "example" and settings.model_backend == "stub":
        parser.error("example requires a configured model")
    runtime = AgentRuntime(settings)
    try:
        if args.command == "example":
            message = "Al abrir un video, se queda detenido en la ruedita y nunca empieza"
            lines = [
                "# Configured-model example",
                "",
                f"Model: `{settings.model_backend}/{settings.model_name}`.",
                "",
                f"Reference rules for the issue: **{rules(message).intent.value}**.",
                "",
            ]
            for text in ("cliente 12", "1234", message):
                reply = runtime.turn("model-example", text)
                lines += [f"**Customer:** {text}", "", f"**Agent:** {reply.message}", ""]
            classification = runtime.snapshot("model-example").values["turn"]["classification"]
            lines += [
                f"Combined pipeline: **{classification['prediction']['intent']}**, path **{classification['path']}**.",
                "",
                "This is a single recorded example, separate from the evaluation splits.",
            ]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(lines) + "\n")
            return
        if args.command == "diagram":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(runtime.graph.get_graph(xray=True).draw_mermaid() + "\n")
            return
        print("Red simulada. Cliente de demostración: cliente 12; factor: 1234. Ctrl-D para salir.")
        while True:
            try:
                message = input("Tú: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if message.strip():
                print("Agente: " + runtime.turn("cli-customer", message).message)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
