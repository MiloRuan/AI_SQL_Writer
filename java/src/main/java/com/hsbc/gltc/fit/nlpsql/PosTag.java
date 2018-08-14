package com.hsbc.gltc.fit.nlpsql;

import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphFactory;
import edu.stanford.nlp.semgraph.SemanticGraphFactory.Mode;
import edu.stanford.nlp.util.MetaClass;


import java.util.ArrayList;
import java.util.List;

/**
 * Created by Milo_Ruan on 2018/8/2.
 */
public class PosTag {
    public static void main(String[] args) {

        List<TaggedWord> tagged = new ArrayList<TaggedWord>();
        tagged.add(new TaggedWord("Table_A", "NN"));
        tagged.add(new TaggedWord("left", "RB"));
        tagged.add(new TaggedWord("join", "VBP"));
        tagged.add(new TaggedWord("Table_B", "NN"));
        tagged.add(new TaggedWord(".", "."));

        String modelPath = DependencyParser.DEFAULT_MODEL;
        DependencyParser parser = DependencyParser.loadFromModelFile(modelPath);

        GrammaticalStructure gs = parser.predict(tagged);
        GrammaticalStructure.Extras extraDependencies = MetaClass.cast("NONE", GrammaticalStructure.Extras.class);

        SemanticGraph deps = SemanticGraphFactory.makeFromTree(gs, Mode.COLLAPSED, extraDependencies, null),
                uncollapseDeps = SemanticGraphFactory.makeFromTree(gs, Mode.BASIC, extraDependencies, null),
                ccDeps = SemanticGraphFactory.makeFromTree(gs, Mode.CCPROCESSED, extraDependencies, null),
                enhancedDeps = SemanticGraphFactory.makeFromTree(gs, Mode.ENHANCED, extraDependencies, null),
                enhancedPlusDeps = SemanticGraphFactory.makeFromTree(gs, Mode.ENHANCED_PLUS_PLUS, extraDependencies, null);

        System.out.println(deps);

    }
}
