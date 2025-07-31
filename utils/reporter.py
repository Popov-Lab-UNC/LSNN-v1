from openmm.unit import *
import h5py

#only need positions from DCD, but cant use dcd so impromptu function:
class PositionsReporter:
    """Saves ligand-only positions to an HDF5 file at specified intervals."""
    def __init__(self, filename, mol_indices, report_interval):
        self.report_interval = report_interval
        self.mol_indices = mol_indices

        self.file = h5py.File(filename, 'w')
        self.file.create_dataset("positions",
                                 maxshape=(None, len(self.mol_indices), 3),
                                 shape=(0, len(self.mol_indices), 3),
                                 dtype=np.float32)

    def __del__(self):
        self.file.close()

    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer)

        # Resize and append ligand positions
        current_length = self.file["positions"].shape[0]
        self.file["positions"].resize(
            (current_length + 1, len(self.mol_indices), 3))
        self.file["positions"][-1] = positions[self.mol_indices]
