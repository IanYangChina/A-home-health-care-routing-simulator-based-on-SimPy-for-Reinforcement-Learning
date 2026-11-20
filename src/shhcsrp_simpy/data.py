"""Scenario loading. Original Excel input is retained, not silently reinterpreted."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np
from openpyxl import load_workbook

# Minutes of service per skill (rows) and demand level (columns).
# Values are from original QL_BWACO/main.py and the 2023 article's Table 1.
SERVICE_MEANS = np.array([
    [25., float('inf'), float('inf')],
    [20., 30., float('inf')],
    [18., 20., 20.],
], dtype=float)
DEFAULT_NURSE_SKILLS = {
    "A": (1, 2, 2, 2, 3, 3, 3),
    "B": (1, 2, 2, 2, 3, 3, 3),
    "C": (1, 2, 2, 2, 3, 3, 3),
    "D": (1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3),
}
DEPOT = (125.0, 125.0, 0.0)

@dataclass(frozen=True)
class Job:
    id: int  # stable zero-based action index; not the workbook's JobNum
    elder_id: int
    demand: int
    position: tuple[float, float, float]
    window_start: float
    window_end: float


@dataclass(frozen=True)
class Scenario:
    name: str
    jobs: tuple[Job, ...]
    nurse_skills: tuple[int, ...]
    preferences: np.ndarray = field(repr=False)
    depot: tuple[float, float, float] = DEPOT
    horizon: float = 480.0
    speed: float = 60.0  # distance units / minute, original main.py
    max_wait: float = 40.0

    def __post_init__(self):
        if not self.jobs or not self.nurse_skills:
            raise ValueError('A scenario needs jobs and nurses.')
        if tuple(j.id for j in self.jobs) != tuple(range(len(self.jobs))):
            raise ValueError('Job IDs must equal contiguous action indices.')
        if self.preferences.ndim != 2 or self.preferences.shape[1] != len(self.nurse_skills):
            raise ValueError('Preference matrix has wrong shape.')
        if max(j.elder_id for j in self.jobs) > self.preferences.shape[0]:
            raise ValueError('Preference matrix has too few elder rows.')


def dataset_dir() -> Path:
    # Installed users can provide --data-dir; repository users get bundled Excel files.
    repo_path = Path(__file__).resolve().parents[2] / 'data'
    return repo_path if repo_path.exists() else Path(__file__).resolve().parent / 'datasets'


def load_scenario(instance: str = 'A', *, root: str | Path | None = None,
                  nurse_skills: Sequence[int] | None = None, job_limit: int | None = None,
                  horizon: float = 480., max_wait: float = 40., speed: float = 60.) -> Scenario:
    """Load an original Excel instance; optionally subset first N jobs for a fast demo.

    Original workbook JobNum indicates appointments-per-elder, NOT a unique job ID.
    Rows are independent service demands; one elder can have multiple rows.
    """
    instance = instance.upper()
    if instance not in 'ABCD' or len(instance) != 1:
        raise ValueError('Instance must be A, B, C or D')
    root = Path(root) if root is not None else dataset_dir()
    path = root / f'Elders_{instance}.xlsx'
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book.active
        data = sheet.iter_rows(values_only=True)
        headers = next(data)
        idx = {col: i for i, col in enumerate(headers) if col is not None}
        needed = ['Indexes', 'JobLevel', 'X', 'Y', 'Z', 'TWB', 'TWE']
        if not all(k in idx for k in needed):
            raise ValueError(f'{path} missing required columns: {needed}')
        jobs = []
        for row in data:
            if row[idx['Indexes']] is None:
                continue
            jobs.append(Job(
                id=len(jobs), elder_id=int(row[idx['Indexes']]),
                demand=int(row[idx['JobLevel']]),
                position=(float(row[idx['X']]), float(row[idx['Y']]), float(row[idx['Z']])),
                window_start=float(row[idx['TWB']]), window_end=float(row[idx['TWE']]),
            ))
            if job_limit is not None and len(jobs) >= job_limit:
                break
    finally:
        book.close()
    skills = tuple(int(s) for s in (nurse_skills or DEFAULT_NURSE_SKILLS[instance]))
    if any(s not in (1, 2, 3) for s in skills):
        raise ValueError('Only skill levels 1, 2, 3 are supported.')
    elder_count = max(j.elder_id for j in jobs)
    pref_fn = {'A': 'initial_preference_AB.csv', 'B': 'initial_preference_AB.csv',
               'C': 'initial_preference_C1.csv', 'D': 'initial_preference_D1.csv'}[instance]
    pth = root / pref_fn
    if pth.exists():
        prefs = np.loadtxt(pth, delimiter=',', dtype=np.float64)
        if prefs.ndim == 1:
            prefs = prefs.reshape((-1, 1))
    else:
        prefs = np.ones((elder_count, len(skills)), dtype=float)
    if prefs.shape[1] < len(skills):
        # Always explicit; avoid silent modulo mappings of nurse columns.
        raise ValueError(f'{pref_fn} has {prefs.shape[1]} nurse columns; requested {len(skills)}')
    # The original model's preferences were per elder and *position* in nurse list.
    prefs = np.asarray(prefs[:elder_count, :len(skills)], dtype=np.float64).copy()
    return Scenario(instance, tuple(jobs), skills, prefs, horizon=horizon,
                    max_wait=max_wait, speed=speed)


def travel_minutes(a: tuple[float, float, float], b: tuple[float, float, float],
                   speed: float) -> float:
    """Legacy distance function from QL_BWACO.get_data, divided by walking speed.

    When XY locations match, vertical distance is weighted by 9.6. Otherwise,
    distance = XY Euclidean distance plus the two floor elevations.
    The latter is an unusual original assumption, retained transparently here.
    """
    horizontal = float(np.hypot(a[0] - b[0], a[1] - b[1]))
    distance = (9.6 * abs(a[2] - b[2])) if horizontal < 1e-12 else (horizontal + a[2] + b[2])
    return distance / speed
