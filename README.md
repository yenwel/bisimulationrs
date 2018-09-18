# bisimulationrs

https://www.ru.is/faculty/luca/PAPERS/algobisimchapter.pdf
https://doc.rust-lang.org/std/sync/mpsc/fn.channel.html
https://arxiv.org/pdf/1311.7635.pdf
http://www.math.unipd.it/~crafa/Pubblicazioni/CrafaRanzatoICALP11.pdf
https://arxiv.org/pdf/1705.08362.pdf
https://docs.rs/carboxyl/0.2.1/carboxyl/
https://ptolemy.eecs.berkeley.edu/eecs20/week4/bisimulation.html
https://arxiv.org/pdf/1101.4223.pdf
http://www.cs.unibo.it/~sangio/DOC_public/history_bis_coind.pdf
http://www.cs.ru.nl/M.Sammartino/publications/ACTA14.pdf
http://coalg.org/cmcs12/papers/00010040.pdf
https://arxiv.org/abs/1506.01170
https://arxiv.org/pdf/1101.4223.pdf
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.8413&rep=rep1&type=pdf
https://www.southampton.ac.uk/~ps1a06/papers/jcss.pdf
https://golem.ph.utexas.edu/category/2008/11/coalgebraically_thinking.html
https://www.georgejpappas.org/papers/ACC06.pdf
https://arxiv.org/pdf/1310.4106.pdf
http://homepages.inf.ed.ac.uk/jeh/Bio-PEPA/TCS_CH.pdf
https://lmcs.episciences.org/1617/pdf
https://arxiv.org/pdf/1509.08563.pdf
http://www.numdam.org/article/ITA_1999__33_4-5_357_0.pdf
http://www.prismmodelchecker.org/manual/ThePRISMLanguage/ProcessAlgebraOperators
http://www.dcs.ed.ac.uk/pepa/
http://www.dcs.ed.ac.uk/pepa/fluidflow.pdf
http://homepages.inf.ed.ac.uk/jeh/TALKS/Qest05.pdf
https://pdfs.semanticscholar.org/d930/9b8202bd47317a3923c54345029715604060.pdf

inputs: stream a, stream b
output: 
- stream<bool> rightsimulatesleft
- stream<bool> arebisimulate
- stream<bool> arebisimilar

just like we can check the the equality (isomorphism) of two objects or sets inductively (or two hypotheses stochastically), the purpose of this library is to check if non-deterministic stuff (streams, cyclical etc) bisumulates each other.

two systems of finite size must be compared in a deterministic manner, an equality algorithm must stop with a time depending on the size of the systems. When comparing two streams you keep on generating a result of the bisumulation, so again a stream. This stream can stop but mustn't.

Instead of checking if two models of a system are equal, we can check if the behaviour of two systems bisimulation each other. This is a trade off, bisimulation will be less accurate but immediate and continuous.

The next step is then to look at solutions for state space explosion. Then add stochastic result (sort of a like student T-test). Try to digg into how this relates to category theory, homotopy type theory, practical application (continuous control, see autotraits in rust, etc.). How does causality factor into this? Benchmarking of processes (pareto, dea etc ) Algebra and coalgebra are dual concepts like real and unreal numbers. Both are logically sound but you can do more with the latter.

One of the main practical solutions is the lifecycle of animals. My first real job was calculating statistics to check the technical and financial productivity of animals. Unlike factory processes or more controlled environment, you have a lot less control over factors (similarly in economics, free market?). Lifecycles of animals are cyclical, interupted by disease of death and can span over a year. There is a lot to be won if we can benchmark the lifecycle processes.