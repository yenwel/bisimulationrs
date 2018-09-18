extern crate carboxyl;
use std::collections::HashMap;


struct LabelledTransitionSystem {
   states : Vec<String>,
   transitions : HashMap<String,(String,String)>
}

impl LabelledTransitionSystem {
    fn new(states : Vec<String>,  transitions : HashMap<String,(String,String)>) -> LabelledTransitionSystem
    {
        LabelledTransitionSystem {
            states,
            transitions
        }
    }
}

fn bisimulates(lts_one : LabelledTransitionSystem, lts_two : LabelledTransitionSystem) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisimilar_for_two_unsimilar_lts() {
        let states_one = 
            vec![
                "born".to_string(), 
                "dead".to_string(), 
                "sick".to_string(), 
                "healthy".to_string()];
        let transitions_one :  HashMap<String,(String,String)> = 
        [("disease".to_string(), ("born".to_string(), "sick".to_string())),
         ("death".to_string(), ("sick".to_string(), "dead".to_string())),
         ("healing".to_string(), ("sick".to_string(),"healthy".to_string()))
        ].iter().cloned().collect();
        let lts_one = LabelledTransitionSystem::new(
            states_one, transitions_one
        );
        let states_two = 
            vec![
                "born".to_string(), 
                "dead".to_string(), 
                "sick".to_string(), 
                "healthy".to_string()];       
        let transitions_two : HashMap<String,(String,String)> = 
        [("disease".to_string(), ("born".to_string(), "sick".to_string())),
         ("death".to_string(), ("sick".to_string(), "dead".to_string())),
         ("healing".to_string(), ("sick".to_string(),"healthy".to_string()))
        ].iter().cloned().collect();
        let lts_two = LabelledTransitionSystem::new(
             states_two, transitions_two
        );
        assert!(bisimulates(lts_one,lts_two));
    }
    
    #[test]
    fn bisimilar_for_two_similar_lts() {
        let states_one = 
            vec![
                "born".to_string(), 
                "dead".to_string(), 
                "sick".to_string(), 
                "healthy".to_string()];
        let transitions_one :  HashMap<String,(String,String)> = 
        [("disease".to_string(), ("born".to_string(), "sick".to_string())),
         ("death".to_string(), ("sick".to_string(), "dead".to_string())),
         ("healing".to_string(), ("sick".to_string(),"healthy".to_string()))
        ].iter().cloned().collect();
        let lts_one = LabelledTransitionSystem::new(
            states_one, transitions_one
        );
        let states_two = 
            vec![
                "born".to_string(), 
                "dead".to_string(), 
                "sick".to_string(), 
                "healthy".to_string()];       
        let transitions_two : HashMap<String,(String,String)> = 
        [("disease".to_string(), ("born".to_string(), "sick".to_string())),
         ("death".to_string(), ("sick".to_string(), "dead".to_string())),
         ("healing".to_string(), ("sick".to_string(),"healthy".to_string()))
        ].iter().cloned().collect();
        let lts_two = LabelledTransitionSystem::new(
             states_two, transitions_two
        );
        assert!(!bisimulates(lts_one,lts_two));
    }
}
