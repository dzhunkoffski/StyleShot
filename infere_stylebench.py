import os
import argparse
import subprocess

def main(opt):
    styles_dir = [
        f for f in os.listdir(opt.styles) if os.path.isdir(os.path.join(opt.styles, f))
    ]
    print(f'All styles: {styles_dir}')
    base_cmd = [
        'python', 'styleshot_image_driven_multiple.py',
        '--content', opt.content,
        '--preprocessor', opt.preprocessor,
        '--prompt', opt.prompt,
        '--device', str(opt.device)
    ]
    if opt.extract_prompt:
        base_cmd += ['--extract_prompt']

    for style in styles_dir:
        print(f'==== PROCESSING STYLE {style} ====')
        cmd = base_cmd + [
            '--style', os.path.join(opt.styles, style),
            '--output', os.path.join(opt.output, style)
        ]
        os.makedirs(os.path.join(opt.output, style), exist_ok=True)
        print(f'=====> CMD: {" ".join(cmd)}')
        subprocess.run(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--styles', type=str)
    parser.add_argument('--content', type=str)
    parser.add_argument("--preprocessor", type=str, default="Contour", choices=["Contour", "Lineart"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--extract_prompt", action='store_true')
    opt = parser.parse_args()
    main(opt)
    print('+++++ FINISHED +++++')
