### Data Flow

Yes/no question -> Preprocess question -> Execute search(es) for congressional reports from congress.gov -> Preprocess text from top results (extract text evidence, some of these results are pretty long) -> Pass as text evidence to BoolQ or other ML model -> yes/no

### UX

Type in yes/no question -> (probably solve Captcha) -> Wait a second... -> Get something like "Yes. According to {source}: {text evidence}"

Start in terminal because this is proof of concept
