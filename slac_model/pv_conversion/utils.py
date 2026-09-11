def get_magnetic_rigidity(energy: float) -> float:
	"""Calculate magnetic rigidity in kG-m for beam energy in eV."""
	return 33.356 * energy / 1e9