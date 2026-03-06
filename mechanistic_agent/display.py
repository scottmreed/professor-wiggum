"""Visual display system for mechanistic agent workflow progress and tool outputs."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Mapping

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.syntax import Syntax
from rich import box


class StepStatus(Enum):
    """Status of workflow steps."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a single step in the mechanistic workflow."""
    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tool_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate step duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def start(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.IN_PROGRESS
        self.start_time = time.time()
    
    def complete(self, tool_output: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as completed with optional tool output."""
        self.status = StepStatus.COMPLETED
        self.end_time = time.time()
        self.tool_output = tool_output
    
    def fail(self, error_message: str) -> None:
        """Mark step as failed with error message."""
        self.status = StepStatus.FAILED
        self.end_time = time.time()
        self.error_message = error_message

    def skip(self, reason: Optional[str] = None) -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.end_time = time.time()
        if reason and not self.error_message:
            self.error_message = reason


@dataclass
class WorkflowProgress:
    """Tracks progress through the mechanistic workflow."""
    steps: List[WorkflowStep] = field(default_factory=list)
    current_step: Optional[str] = None
    start_time: Optional[float] = None
    console: Console = field(default_factory=Console)
    stop_requested: bool = False
    stop_reason: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default workflow steps."""
        if not self.steps:
            self.steps = [
                WorkflowStep("balance_analysis", "Check atomic balance between reactants and products"),
                WorkflowStep("functional_groups", "Analyze functional groups and reactive sites"),
                WorkflowStep("initial_conditions", "Assess initial pH guidance and additives"),
                WorkflowStep("missing_reagents", "Predict missing reagents if needed"),
                WorkflowStep("atom_mapping", "Attempt atom mapping between reactants and products"),
                WorkflowStep("reaction_type_mapping", "Map reaction to taxonomy type"),
                WorkflowStep("ph_recommendation", "Estimate optimal pH range"),
                WorkflowStep("intermediates", "Propose likely intermediate species"),
                WorkflowStep("mechanism_synthesis", "Generate reaction mechanism steps"),
            ]
    
    def start_workflow(self) -> None:
        """Mark workflow as started."""
        self.start_time = time.time()
        self.stop_requested = False
        self.stop_reason = None
        self.console.print("\n[bold blue]🧪 Mechanistic Agent Workflow Started[/bold blue]\n")
    
    def start_step(self, step_name: str) -> None:
        """Start a specific workflow step."""
        self.current_step = step_name
        step = self.get_step(step_name)
        if step:
            step.start()
            self._display_step_start(step)
    
    def complete_step(self, step_name: str, tool_output: Optional[Dict[str, Any]] = None) -> None:
        """Complete a specific workflow step."""
        step = self.get_step(step_name)
        if step:
            step.complete(tool_output)
            self._display_step_complete(step)
    
    def fail_step(self, step_name: str, error_message: str) -> None:
        """Mark a workflow step as failed."""
        step = self.get_step(step_name)
        if step:
            step.fail(error_message)
            self._display_step_failed(step)

    def request_stop(self, reason: Optional[str] = None) -> None:
        """Record that execution should stop."""
        reason_text = (reason or "User requested stop").strip()
        if self.stop_requested:
            self.stop_reason = reason_text
            return
        self.stop_requested = True
        self.stop_reason = reason_text
        self.console.print(f"[red]⛔ Stop requested: {reason_text}[/red]")

    def get_step(self, step_name: str) -> Optional[WorkflowStep]:
        """Get a workflow step by name."""
        return next((step for step in self.steps if step.name == step_name), None)

    def skip_pending_steps(self, reason: Optional[str] = None) -> None:
        """Mark any remaining pending or in-progress steps as skipped."""
        for step in self.steps:
            if step.status in {StepStatus.PENDING, StepStatus.IN_PROGRESS}:
                step.skip(reason)
        if reason:
            self.console.print(f"[yellow]⚠️ Pending steps skipped: {reason}[/yellow]")
        else:
            self.console.print("[yellow]⚠️ Pending steps skipped.[/yellow]")
    
    def _display_step_start(self, step: WorkflowStep) -> None:
        """Display when a step starts."""
        self.console.print(f"[yellow]⏳ {step.description}...[/yellow]")
    
    def _display_step_complete(self, step: WorkflowStep) -> None:
        """Display when a step completes."""
        duration = f" ({step.duration:.2f}s)" if step.duration else ""
        self.console.print(f"[green]✅ {step.description} completed{duration}[/green]")
        
        if step.tool_output:
            self._display_tool_output(step.name, step.tool_output)
    
    def _display_step_failed(self, step: WorkflowStep) -> None:
        """Display when a step fails."""
        self.console.print(f"[red]❌ {step.description} failed: {step.error_message}[/red]")
    
    def _display_tool_output(self, step_name: str, tool_output: Dict[str, Any]) -> None:
        """Display formatted tool output."""
        if step_name == "balance_analysis":
            self._display_balance_analysis(tool_output)
        elif step_name == "atom_mapping":
            self._display_atom_mapping(tool_output)
        elif step_name == "initial_conditions":
            self._display_initial_conditions(tool_output)
        elif step_name == "missing_reagents":
            self._display_missing_reagents(tool_output)
        elif step_name == "ph_recommendation":
            self._display_ph_recommendation(tool_output)
        elif step_name == "functional_groups":
            self._display_functional_groups(tool_output)
        elif step_name == "intermediates":
            self._display_intermediates(tool_output)
        else:
            self._display_generic_tool_output(tool_output)
    
    def _display_balance_analysis(self, output: Dict[str, Any]) -> None:
        """Display atomic balance analysis results."""
        if not isinstance(output, dict):
            return

        rdkit_data = output.get("rdkit") if "rdkit" in output else output
        if not isinstance(rdkit_data, dict):
            return

        def _format_counter(counter: Optional[Dict[str, int]]) -> str:
            if not counter:
                return "—"
            return ", ".join(f"{element}: {amount}" for element, amount in sorted(counter.items()))

        def _render_balance_table(label: str, data: Dict[str, Any]) -> None:
            reactant_counts = data.get("reactant_counts") or {}
            product_counts = data.get("product_counts") or {}
            elements = sorted(set(reactant_counts) | set(product_counts))
            if not elements and "starting_materials" in data:
                pattern = re.compile(r"[A-Z][a-z]?")
                molecules = data.get("starting_materials", []) + data.get("products", [])
                extracted: set[str] = set()
                for mol in molecules:
                    formula = mol.get("formula", "") if isinstance(mol, dict) else ""
                    extracted.update(match.group(0) for match in pattern.finditer(formula))
                elements = sorted(extracted)

            table = Table(title=f"Atomic Balance Analysis — {label}", box=box.ROUNDED)
            table.add_column("Element", style="cyan")
            table.add_column("Reactants", justify="right")
            table.add_column("Products", justify="right")
            table.add_column("Δ", justify="right")

            for element in elements:
                reactant = int(reactant_counts.get(element, 0))
                product = int(product_counts.get(element, 0))
                delta = product - reactant
                style = "green" if delta == 0 else "red" if delta < 0 else "yellow"
                table.add_row(
                    element,
                    str(reactant),
                    str(product),
                    Text(f"{delta:+d}", style=style),
                )

            self.console.print(table)

            if data.get("balanced"):
                self.console.print("[green]✅ Reaction is atomically balanced[/green]")
            else:
                deficit = data.get("deficit") or {}
                surplus = data.get("surplus") or {}
                if deficit:
                    self.console.print(
                        f"[red]❌ Missing atoms: {_format_counter(deficit)}[/red]"
                    )
                if surplus:
                    self.console.print(
                        f"[yellow]⚠️ Excess atoms: {_format_counter(surplus)}[/yellow]"
                    )

        if rdkit_data:
            _render_balance_table("RDKit", rdkit_data)

    def _display_atom_mapping(self, output: Dict[str, Any]) -> None:
        """Display the atom mapping guidance returned by the LLM tool."""
        if not isinstance(output, dict):
            self._display_generic_tool_output(output)
            return

        stoichiometry = output.get("stoichiometry")
        if isinstance(stoichiometry, dict):
            table = Table(title="Stoichiometry Summary", box=box.ROUNDED)
            table.add_column("Set", style="cyan")
            table.add_column("Atom counts", style="white")
            for label in ("reactants", "products"):
                counts = stoichiometry.get(label, {})
                if isinstance(counts, dict):
                    text = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "—"
                else:
                    text = str(counts)
                table.add_row(label.title(), text)
            self.console.print(table)

        deficit = output.get("deficit")
        if isinstance(deficit, dict) and deficit:
            deficit_text = ", ".join(f"{k}: {v}" for k, v in deficit.items())
            self.console.print(f"[red]❗ Missing atoms relative to reactants: {deficit_text}[/red]")

        surplus = output.get("surplus")
        if isinstance(surplus, dict) and surplus:
            surplus_text = ", ".join(f"{k}: {v}" for k, v in surplus.items())
            self.console.print(f"[yellow]⚠️  Excess atoms relative to products: {surplus_text}[/yellow]")

        model_name = output.get("mapping_model")
        if model_name:
            self.console.print(f"[dim]Atom mapping model: {model_name}[/dim]")

        error_message = output.get("error")
        if error_message:
            self.console.print(f"[red]⚠️  Atom mapping unavailable: {error_message}[/red]")

        mapping = output.get("llm_response")
        if isinstance(mapping, dict):
            mapped_atoms = mapping.get("mapped_atoms")
            if isinstance(mapped_atoms, list) and mapped_atoms:
                mapping_table = Table(title="Mapped Atoms", box=box.ROUNDED)
                mapping_table.add_column("Product atom", style="magenta")
                mapping_table.add_column("Source", style="cyan")
                mapping_table.add_column("Notes", style="white")
                for entry in mapped_atoms:
                    if not isinstance(entry, dict):
                        continue
                    product_atom = str(entry.get("product_atom", "-"))
                    source = entry.get("source")
                    if isinstance(source, dict):
                        parts: List[str] = []
                        molecule_index = source.get("molecule_index")
                        if molecule_index is not None:
                            parts.append(f"mol {molecule_index}")
                        atom_index = source.get("atom_index")
                        if atom_index is not None:
                            parts.append(f"atom {atom_index}")
                        smiles = source.get("smiles")
                        if smiles:
                            parts.append(smiles)
                        source_text = ", ".join(parts) if parts else json.dumps(source)
                    else:
                        source_text = str(source)
                    notes = entry.get("notes") or ""
                    mapping_table.add_row(product_atom, source_text or "-", notes)
                self.console.print(mapping_table)
            elif mapped_atoms is None:
                self.console.print("[yellow]⚠️  LLM did not provide a confident atom mapping.[/yellow]")

            unmapped = mapping.get("unmapped_atoms")
            if isinstance(unmapped, list) and unmapped:
                bullets = "\n".join(f"  • {item}" for item in unmapped)
                self.console.print(f"[yellow]Unmapped atoms or notes:\n{bullets}")

            confidence = mapping.get("confidence")
            if confidence:
                self.console.print(f"[dim]Reported confidence: {confidence}[/dim]")

            reasoning = mapping.get("reasoning")
            if reasoning:
                self.console.print(f"[cyan]Rationale:[/cyan] {reasoning}")

        raw = output.get("raw_response")
        if raw and not isinstance(mapping, dict):
            self.console.print(Syntax(raw, "json", theme="monokai", word_wrap=True))

        guidance = output.get("guidance")
        if guidance:
            self.console.print(f"[dim]{guidance}[/dim]")

    def _display_initial_conditions(self, output: Dict[str, Any]) -> None:
        """Render initial pH guidance and additive recommendations."""
        if not isinstance(output, dict):
            return

        environment = output.get("environment") or output.get("environment_bias")
        rep_ph_raw = output.get("representative_ph")
        ph_range = output.get("ph_range")
        justification = output.get("justification")
        warnings = output.get("warnings")

        details: List[str] = []
        if environment:
            details.append(f"[bold]Preferred environment:[/bold] {environment}")

        try:
            if rep_ph_raw is not None:
                rep_ph_val = float(rep_ph_raw)
                details.append(f"[bold]Representative pH:[/bold] {rep_ph_val:.2f}")
        except (TypeError, ValueError):  # pragma: no cover - defensive
            details.append(f"[bold]Representative pH:[/bold] {rep_ph_raw}")

        if isinstance(ph_range, (list, tuple)) and len(ph_range) == 2:
            try:
                lower = float(ph_range[0])
                upper = float(ph_range[1])
                details.append(f"[bold]Suggested pH span:[/bold] {lower:.2f} – {upper:.2f}")
            except (TypeError, ValueError):
                details.append(f"[bold]Suggested pH span:[/bold] {ph_range}")

        if justification:
            details.append(f"[dim]{justification}[/dim]")

        if details:
            panel = Panel("\n".join(details), title="Initial Condition Guidance", border_style="blue")
            self.console.print(panel)

        def _render_candidates(title: str, entries: Any) -> None:
            if not isinstance(entries, (list, tuple)) or not entries:
                return

            table = Table(title=title, box=box.SIMPLE_HEAVY)
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("SMILES", style="magenta", no_wrap=True)
            table.add_column("Justification", style="green")

            for entry in entries:
                if isinstance(entry, Mapping):
                    name = str(entry.get("name") or "—")
                    smiles = str(entry.get("smiles") or "")
                    reason = str(entry.get("justification") or entry.get("reason") or "")
                else:
                    name = str(entry)
                    smiles = ""
                    reason = ""
                table.add_row(name, smiles, reason)

            self.console.print(table)

        _render_candidates("Acid Supports", output.get("acid_candidates"))
        _render_candidates("Base Supports", output.get("base_candidates"))
        _render_candidates("Buffer Suggestions", output.get("buffer_suggestions"))

        if warnings:
            warning_lines: List[str]
            if isinstance(warnings, (list, tuple)):
                warning_lines = [f"• {item}" for item in warnings if item]
            else:
                warning_lines = [f"• {warnings}"]
            if warning_lines:
                self.console.print("[yellow]Warnings:\n" + "\n".join(warning_lines) + "[/yellow]")

    def _display_missing_reagents(self, output: Dict[str, Any]) -> None:
        """Display missing reagent predictions."""
        if not isinstance(output, dict):
            return
        
        atom_delta = output.get("atom_delta", {})
        suggestions = output.get("suggested_reagents", [])
        status = output.get("status")
        message = output.get("message")
        error = output.get("error")
        parse_error = output.get("parse_error")
        invalid = output.get("invalid_reagents") or []
        abort_flag = bool(output.get("should_abort_mechanism"))

        if atom_delta:
            panel = Panel(
                f"[bold]Missing Atoms:[/bold] {', '.join(f'{k}: {v}' for k, v in atom_delta.items())}\n"
                f"[bold]Suggested Reagents:[/bold] {', '.join(suggestions) if suggestions else 'None'}",
                title="Missing Reagent Analysis",
                border_style="yellow"
            )
            self.console.print(panel)

        if status:
            colour = "green" if status == "success" else "red" if status == "failed" else "blue"
            self.console.print(f"[{colour}]Status: {status}[/]")

        if message:
            self.console.print(f"[cyan]{message}[/cyan]")

        if error:
            self.console.print(f"[red]Error: {error}[/red]")

        if parse_error:
            self.console.print(f"[red]Response parsing issue: {parse_error}[/red]")

        if invalid:
            invalid_lines = [
                f"  • {item.get('reagent', '?')} ({item.get('error', 'invalid')})"
                for item in invalid
                if isinstance(item, dict)
            ]
            if invalid_lines:
                self.console.print("[red]Invalid reagents returned:\n" + "\n".join(invalid_lines))

        if abort_flag:
            self.console.print("[bold red]Mechanism generation halted: unable to balance reaction.[/bold red]")
    
    def _display_ph_recommendation(self, output: Dict[str, Any]) -> None:
        """Display pH recommendation results."""
        if not isinstance(output, dict):
            return
        
        source = output.get("source", "unknown")
        
        if source == "user":
            ph = output.get("provided_ph")
            self.console.print(f"[blue]📊 Using provided pH: {ph}[/blue]")
        elif source == "dimorphite_dl":
            self.console.print("[blue]📊 pH analysis using Dimorphite-DL (see detailed profiles)[/blue]")
        elif source == "heuristic":
            ph_range = output.get("recommended_range")
            rationale = output.get("rationale", "")
            if ph_range:
                self.console.print(f"[blue]📊 Recommended pH range: {ph_range[0]:.1f} - {ph_range[1]:.1f}[/blue]")
                self.console.print(f"[dim]Rationale: {rationale}[/dim]")
    
    def _display_functional_groups(self, output: Dict[str, Any]) -> None:
        """Display functional group analysis."""
        if not isinstance(output, dict):
            return
        
        functional_groups = output.get("functional_groups", {})
        
        if functional_groups:
            table = Table(title="Functional Group Analysis", box=box.ROUNDED)
            table.add_column("SMILES", style="cyan", max_width=20)
            table.add_column("Functional Groups", style="green")
            
            for smiles, groups in functional_groups.items():
                group_text = ", ".join(f"{name}: {count}" for name, count in groups.items() if count > 0)
                if not group_text:
                    group_text = "None detected"
                table.add_row(smiles[:20] + "..." if len(smiles) > 20 else smiles, group_text)
            
            self.console.print(table)
    
    def _display_intermediates(self, output: Dict[str, Any]) -> None:
        """Display intermediate predictions."""
        if not isinstance(output, dict):
            return
        
        intermediates = output.get("intermediates", [])
        
        if intermediates:
            table = Table(title="Proposed Intermediates", box=box.ROUNDED)
            table.add_column("Type", style="cyan")
            table.add_column("SMILES", style="green")
            table.add_column("Notes", style="dim")
            
            for intermediate in intermediates:
                table.add_row(
                    intermediate.get("type", "unknown"),
                    intermediate.get("smiles", ""),
                    intermediate.get("note", "")
                )
            
            self.console.print(table)
        else:
            self.console.print("[dim]No intermediates proposed[/dim]")
    
    def _display_generic_tool_output(self, output: Dict[str, Any]) -> None:
        """Display generic tool output as formatted JSON."""
        if output:
            json_str = json.dumps(output, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title="Tool Output", border_style="blue"))
    
    def display_progress_summary(self) -> None:
        """Display a summary of workflow progress."""
        completed = sum(1 for step in self.steps if step.status == StepStatus.COMPLETED)
        total = len(self.steps)
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Workflow Progress", total=100)
            progress.update(task, completed=int((completed / total) * 100))
        
        # Create status table
        table = Table(title="Workflow Status", box=box.ROUNDED)
        table.add_column("Step", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        
        for step in self.steps:
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }.get(step.status, "❓")
            
            duration = f"{step.duration:.2f}s" if step.duration else "-"
            
            table.add_row(
                step.description,
                status_icon,
                duration
            )
        
        self.console.print(table)
    
    def display_workflow_complete(self) -> None:
        """Display workflow completion summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        completed_steps = sum(1 for step in self.steps if step.status == StepStatus.COMPLETED)
        failed_steps = sum(1 for step in self.steps if step.status == StepStatus.FAILED)
        
        self.console.print(f"\n[bold green]🎉 Workflow Complete![/bold green]")
        self.console.print(f"[dim]Completed: {completed_steps} steps | Failed: {failed_steps} steps | Total time: {total_time:.2f}s[/dim]\n")


def create_workflow_progress() -> WorkflowProgress:
    """Create a new workflow progress tracker."""
    return WorkflowProgress()


__all__ = ["WorkflowProgress", "WorkflowStep", "StepStatus", "create_workflow_progress"]
