"""Boundary tests for delegation, immutable inputs and error propagation."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from x2_recovery import reproduce


class ReproduceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'input';self.source.mkdir()
        (self.source/'policy_final.zip').write_bytes(b'original-frozen-input')

    def args(self,operation='run'):
        return argparse.Namespace(operation=operation,training_run=self.source,output=self.root/'new output',
            expected_checkpoint_sha256='a'*64,seed=221030,seeds=[1,2,3,4,5],episode_wall_seconds=180.)

    def prepared(self):
        file=self.source/'policy_final.zip'
        return dict(controller='reference_residual',training_run=str(self.source),
                    input_sha256={file.name:hashlib.sha256(file.read_bytes()).hexdigest()}),{file.name:file}

    def test_refuses_existing_and_nested_output_before_loading(self):
        with patch.object(reproduce,'checked_inputs') as prepare:
            for out in (self.source,self.source/'nested',self.root):
                a=self.args();a.output=out
                with self.assertRaises(ValueError):reproduce.execute(a)
            prepare.assert_not_called()
        self.assertEqual((self.source/'policy_final.zip').read_bytes(),b'original-frozen-input')

    def test_check_cannot_spawn_or_step(self):
        with patch.object(reproduce,'checked_inputs',return_value=self.prepared()) as check, \
             patch.object(reproduce.subprocess,'run') as child:
            self.assertEqual(reproduce.execute(self.args('check')),0)
            child.assert_not_called();check.assert_called_once()

    def test_run_copies_input_and_propagates_failure(self):
        def child(command,**kwargs):
            self.assertIsInstance(command,list)
            self.assertIn('control-reload',command)
            self.assertNotIn('control-block',command)
            self.assertNotIn('learn',command)
            copied=Path(command[command.index('--run-dir')+1])
            self.assertNotEqual(copied,self.source)
            self.assertEqual((copied/'policy_final.zip').read_bytes(),b'original-frozen-input')
            self.assertFalse(kwargs.get('shell',False))
            return subprocess.CompletedProcess(command,7)
        with patch.object(reproduce,'checked_inputs',return_value=self.prepared()), \
             patch.object(reproduce.subprocess,'run',side_effect=child):
            self.assertEqual(reproduce.execute(self.args()),7)
        self.assertEqual(list(self.source.iterdir()),[self.source/'policy_final.zip'])

    def test_evaluate_requires_five_distinct_seeds_before_loading(self):
        with patch.object(reproduce,'checked_inputs') as check:
            a=self.args('evaluate');a.seeds=[1,1,2,3,4]
            with self.assertRaises(ValueError):reproduce.execute(a)
            check.assert_not_called()

    def test_changed_copy_rejected_before_subprocess(self):
        report,files=self.prepared();report['input_sha256']['policy_final.zip']='0'*64
        with patch.object(reproduce,'checked_inputs',return_value=(report,files)), \
             patch.object(reproduce.subprocess,'run') as child:
            with self.assertRaises(ValueError):reproduce.execute(self.args())
            child.assert_not_called()


if __name__=='__main__':unittest.main()
